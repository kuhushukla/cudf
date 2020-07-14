package ai.rapids.cudf;

import jdk.nashorn.internal.ir.IfNode;

import java.util.Optional;

public class ListColumnVector extends BaseColumnVector {
  ListColumnVector childLcv = null;
  BaseColumnVector.OffHeapState offHeap = null;

  public ListColumnVector(ListColumnVector childLcv, BaseColumnVector.OffHeapState offHeap) {
    this.childLcv = childLcv;
    this.type = DType.fromNative(getNativeTypeId(offHeap.getViewHandle()));
    this.offHeap = offHeap;
    System.out.println("KUHU LCV viewHandle=" + offHeap.getViewHandle());
  }

  public ListColumnVector(DType type, int rows, DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets, ListColumnVector childLcv) {
    this.type = type;
    this.childLcv = childLcv;
    if (this.type != DType.EMPTY) {
      int newRows = this.type == DType.LIST ? (int)offsets.length/DType.INT32.getSizeInBytes(): rows;
      this.offHeap = new BaseColumnVector.OffHeapState(type, newRows, Optional.empty(), data, valid, offsets, childLcv);
      System.out.println("KUHU LCV viewHandle=" + offHeap.getViewHandle());
      if (this.offHeap.getOffsets() != null) {
        System.out.println("KUHU ListColumnVector type =" + this.type + "offsets/int32=" +offsets.length/DType.INT32.getSizeInBytes() +" ohs="
        +this.offHeap.getOffsets().length/DType.INT32.getSizeInBytes());
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
