import unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from test.amd.hw.test_cdna_vop3 import run_cdna_lanes

class TestCDNADSSwizzle(unittest.TestCase):
  def test_modes(self):
    # the source lane of lanes 0-7 for each mode of the ISA pseudocode
    for offset, src in {0x041f: [1, 0, 3, 2, 5, 4, 7, 6],                  # bitmask: xor 1
                        0x00a0: [5]*8,                                     # bitmask: or 5, broadcast
                        0x801b: [3, 2, 1, 0, 7, 6, 5, 4],                  # quad permute: reversed
                        0xc020: [1, 2, 3, 4, 5, 6, 7, 8],                  # rotate by 1
                        0xe000: [0, 16, 8, 24, 4, 20, 12, 28]}.items():    # fft: bit reversed
      with self.subTest(offset=hex(offset)):
        out = run_cdna_lanes([cdna.ds_swizzle_b32(vdst=cdna.v[2], addr=cdna.v[255], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0)])
        self.assertEqual(out[:8], src)

def _tr_elem(lane: int, elem: int) -> int:
  # lane id in the low bits, plus the 16-lane group, so a gather from the wrong group cannot pass
  return ((lane & 15) + 3 * elem + (lane >> 4)) & 0xffff

def _tr_expected() -> list[list[int]]:
  """Hardware-verified lane map: lane l halfword n reads lane ((l & ~0xf) | ((l >> 2) & 3)) + 4*n, halfword l & 3."""
  words = [[0] * 64 for _ in range(2)]
  for lane in range(64):
    for halfword in range(4):
      src_lane = ((lane & ~0xf) | ((lane >> 2) & 3)) + 4 * halfword
      val = _tr_elem(src_lane, lane & 3)
      words[halfword // 2][lane] |= val << (16 * (halfword & 1))
  return words

def _tr_program(offset: int = 0, acc: int = 0, vdst: int = 2, partial_exec: bool = False) -> list:
  code = [cdna.v_mov_b32_e32(cdna.v[2], 0xdeadbeef), cdna.v_mov_b32_e32(cdna.v[3], 0xdeadbeef),
          cdna.v_mov_b32_e32(cdna.v[4], 0), cdna.v_mov_b32_e32(cdna.v[5], 0),
          cdna.v_lshlrev_b32_e32(cdna.v[0], 3, cdna.v[255]),
          # v[8] = (lane & 15) + (lane >> 4), the part of the element value that does not depend on the element index
          cdna.v_and_b32_e32(cdna.v[8], 15, cdna.v[255]), cdna.v_lshrrev_b32_e32(cdna.v[7], 4, cdna.v[255]),
          cdna.v_add_u32_e32(cdna.v[8], cdna.v[7], cdna.v[8])]
  for elem in range(4):
    code.append(cdna.v_add_u32_e32(cdna.v[9], 3 * elem, cdna.v[8]) if elem else cdna.v_mov_b32_e32(cdna.v[9], cdna.v[8]))
    dword, shift = divmod(elem * 16, 32)
    if shift == 0: code.append(cdna.v_or_b32_e32(cdna.v[4 + dword], cdna.v[9], cdna.v[4 + dword]))
    else: code.append(cdna.v_lshl_or_b32(cdna.v[4 + dword], cdna.v[9], shift, cdna.v[4 + dword]))
  code += [cdna.ds_write_b64(addr=cdna.v[0], data0=cdna.v[4:5], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0)]
  if partial_exec: code += [cdna.s_mov_b32(cdna.EXEC_LO, 0xffff), cdna.s_mov_b32(cdna.EXEC_HI, 0)]
  code.append(cdna.ds_read_b64_tr_b16(vdst=cdna.v[vdst:vdst + 1], addr=cdna.v[0], offset0=offset & 255, offset1=offset >> 8, acc=acc))
  if partial_exec: code += [cdna.s_mov_b32(cdna.EXEC_LO, 0xffffffff), cdna.s_mov_b32(cdna.EXEC_HI, 0xffffffff)]
  code.append(cdna.s_waitcnt(0))
  if acc: code += [cdna.v_accvgpr_read(cdna.v[vdst + i], cdna.v[vdst + i]) for i in range(2)]
  return code

class TestCDNADSReadTRB16(unittest.TestCase):
  def test_spec_lanes(self):
    # frozen lanes, including lane 16, so the Python spec cannot drift into "every group looks the same"
    expected = _tr_expected()
    self.assertEqual(expected[0][0], 0x00040000)
    self.assertEqual(expected[1][0], 0x000C0008)
    self.assertEqual(expected[0][1], 0x00070003)
    self.assertEqual(expected[1][1], 0x000F000B)
    self.assertEqual(expected[0][16], 0x00050001)

  def test_transpose(self):
    self.assertEqual([run_cdna_lanes(_tr_program(), out_reg=2 + w) for w in range(2)], _tr_expected())

  def test_exec_masks_destination_lanes(self):
    got = [run_cdna_lanes(_tr_program(partial_exec=True), out_reg=2 + w) for w in range(2)]
    expected = _tr_expected()
    for w in range(2):
      self.assertEqual(got[w][:16], expected[w][:16])
      self.assertEqual(got[w][16:], [0xdeadbeef] * 48)

  def test_offset_acc_and_vdst_addr_alias(self):
    expected = _tr_expected()
    for kwargs in ({'offset': 0x108}, {'acc': 1}, {'vdst': 0}):
      with self.subTest(**kwargs):
        vdst = kwargs.get('vdst', 2)
        self.assertEqual([run_cdna_lanes(_tr_program(**kwargs), out_reg=vdst + w) for w in range(2)], expected)


if __name__ == "__main__":
  unittest.main()
