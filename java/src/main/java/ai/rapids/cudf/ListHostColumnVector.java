package ai.rapids.cudf;

public class ListHostColumnVector extends BaseHostColumnVector {
  ListHostColumnVector childLcv = null;
  BaseHostColumnVector.OffHeapState offHeap = null;

  public ListHostColumnVector(DType type, HostMemoryBuffer data, HostMemoryBuffer valid, HostMemoryBuffer offsets) {
    this.type = type;
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
}
