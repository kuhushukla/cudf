package ai.rapids.cudf;

import java.util.Optional;

public class ListColumnVector extends BaseColumnVector {
  ListColumnVector childLcv = null;
  BaseColumnVector.OffHeapState offHeap = null;

  public ListColumnVector(ListColumnVector childLcv, BaseColumnVector.OffHeapState offHeap) {
    this.childLcv = childLcv;
    this.offHeap = offHeap;
  }

  public ListColumnVector(long columnHandle) {
    this.type = DType.fromNative(getNativeTypeId(columnHandle));
    System.out.println("KUHU ListColumnVector type =" + this.type);
    if (this.type != DType.EMPTY) {
      this.offHeap = new OffHeapState(columnHandle);
      if (this.offHeap.getOffsets() != null) {
        System.out.println("KUHU ListColumnVector offsets =" + this.offHeap.getOffsets().address + " len =" +
            this.offHeap.getOffsets().length);
      }
      if (this.type == DType.LIST) {
        this.childLcv = getListChildColumnView(columnHandle);
      } else {
        this.childLcv = null;
      }
    }
  }

  public ListColumnVector(DType type, int rows, DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets) {
    this.type = type;
    System.out.println("KUHU ListColumnVector type =" + this.type);
    if (this.type != DType.EMPTY) {
      this.offHeap = new BaseColumnVector.OffHeapState(type, rows, Optional.empty(), data, valid, offsets);
      if (this.offHeap.getOffsets() != null) {
        System.out.println("KUHU ListHostColumnVector offsets =" + this.offHeap.getOffsets().address + " len =" +
            this.offHeap.getOffsets().length);
      }
//      this.childLcv = getListChildColumnView(columnHandle);
    }
  }

  protected ListColumnVector getListChildColumnView(long address) {
    long value = getChildColumnView(address);
    System.out.println("KUHU getListChildColumnView value =" + value);
    return new ListColumnVector(value);
  }
}
