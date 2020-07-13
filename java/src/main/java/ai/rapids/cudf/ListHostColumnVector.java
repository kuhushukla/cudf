package ai.rapids.cudf;

public class ListHostColumnVector extends BaseHostColumnVector {
  ListHostColumnVector childLcv = null;

  public ListHostColumnVector(DType type, int rows, HostMemoryBuffer data, HostMemoryBuffer valid, HostMemoryBuffer offsets) {
    this.type = type;
    this.rows = rows;
    System.out.println("KUHU ListHostColumnVector type =" + this.type);
    if (this.type != DType.EMPTY) {
      this.offHeap = new BaseHostColumnVector.OffHeapState(data, valid, offsets);
      if (this.offHeap.offsets != null) {
        System.out.println("KUHU ListHostColumnVector offsets =" + this.offHeap.offsets.address + " len =" +
            this.offHeap.offsets.length);
      }
//      this.childLcv = getListChildColumnView(columnHandle);
    }
  }

  public ListHostColumnVector(ListHostColumnVector childLcv, BaseHostColumnVector.OffHeapState offHeap) {
    this.childLcv = childLcv;
    this.offHeap = offHeap;
  }
  @Override
  protected BaseHostColumnVector getChild() {
    return childLcv;
  }
}
