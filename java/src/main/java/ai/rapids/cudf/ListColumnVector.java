package ai.rapids.cudf;

import java.util.Optional;

public class ListColumnVector extends BaseColumnVector {
  ListColumnVector childLcv = null;
  BaseColumnVector.OffHeapState offHeap = null;

  public ListColumnVector(ListColumnVector childLcv, BaseColumnVector.OffHeapState offHeap) {
    this.childLcv = childLcv;
    this.offHeap = offHeap;
  }

  public ListColumnVector(DType type, int rows, DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets, ListColumnVector childLcv) {
    this.type = type;
    this.childLcv = childLcv;
    System.out.println("KUHU ListColumnVector type =" + this.type);
    if (this.type != DType.EMPTY) {
      this.offHeap = new BaseColumnVector.OffHeapState(type, rows, Optional.empty(), data, valid, offsets, childLcv);
      if (this.offHeap.getOffsets() != null) {
        System.out.println("KUHU ListHostColumnVector offsets =" + this.offHeap.getOffsets().address + " len =" +
            this.offHeap.getOffsets().length);
      }
//      this.childLcv = getListChildColumnView(columnHandle);
    }
  }
  @Override
  protected BaseColumnVector getChild() {
    return childLcv;
  }
}
