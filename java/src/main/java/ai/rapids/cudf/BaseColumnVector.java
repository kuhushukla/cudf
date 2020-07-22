package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Optional;

public abstract class BaseColumnVector {

  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);
  protected DType type;
  protected long rows;
  protected abstract BaseColumnVector getChild();
  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  final class OffHeapState extends MemoryCleaner.Cleaner {
    // This must be kept in sync with the native code
    public static final long UNKNOWN_NULL_COUNT = -1;

    public long getColumnHandle() {
      return columnHandle;
    }

    private long columnHandle;
    private long viewHandle = 0;
    private BaseDeviceMemoryBuffer data;
    private BaseDeviceMemoryBuffer valid;
    private BaseDeviceMemoryBuffer offsets;

    public void setLcv(ListColumnVector lcv) {
      this.lcv = lcv;
    }

    private ListColumnVector lcv;

    /**
     * Make a column form an existing cudf::column *.
     */
    public OffHeapState(long columnHandle) {
      this.columnHandle = columnHandle;
      data = getNativeDataPointer();
      valid = getNativeValidPointer();
      offsets = getNativeOffsetsPointer();
    }

    public OffHeapState(long viewHandle, boolean throwAway) {
      this.columnHandle = viewHandle;
      this.viewHandle = viewHandle;
      data = getNativeDataPointer();
      valid = getNativeValidPointer();
      offsets = getNativeOffsetsPointer();
    }
    /**
     * Create a cudf::column_view from device side data.
     */
    public OffHeapState(DType type, int rows, Optional<Long> nullCount,
                        DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets,ListColumnVector lcv) {
      assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
          || !nullCount.isPresent();
      int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
      this.lcv = lcv;
      if (rows == 0) {
        this.columnHandle = makeEmptyCudfColumn(type.nativeId);
      } else {
        long cd = data == null ? 0 : data.address;
        long cdSize = data == null ? 0 : data.length;
        long od = offsets == null ? 0 : offsets.address;
        long vd = valid == null ? 0 : valid.address;
        if (type == DType.LIST) {
          this.viewHandle = makeCudfColumnView(type.nativeId, cd, cdSize, od, vd, nc, rows, lcv.offHeap.getViewHandle());
//          System.out.println(rows+"KUHU NEW CV "+new ColumnVector(viewHandle).offHeap.getOffsets().length);
        } else {
//          HostMemoryBuffer hvalid = HostMemoryBuffer.allocate(valid.length);
//          hvalid.copyFromDeviceBuffer(valid);
//          byte[] tmp = new byte[(int)hvalid.length];
//          System.out.println("KUHU TMP HVALID makecudfview==========" + valid.length + " " + hvalid.length);
//          hvalid.getBytes(tmp,0,0,tmp.length);
//          for (int i = 0; i < tmp.length; i++) {
//            String hex = String.format("%x", (int) tmp[i]);
//            System.out.print(hex + " ");
//          }
          this.viewHandle = makeCudfColumnView(type.nativeId, cd, cdSize, od, vd, nc, rows, 0l);
        }
      }
    }

    /**
     * Create a cudf::column_view from c`ontiguous device side data.
     */
    public OffHeapState(long viewHandle, DeviceMemoryBuffer contiguousBuffer) {
      assert viewHandle != 0;
      this.viewHandle = viewHandle;

      data = contiguousBuffer.sliceFrom(getNativeDataPointer());
      valid = contiguousBuffer.sliceFrom(getNativeValidPointer());
      offsets = contiguousBuffer.sliceFrom(getNativeOffsetsPointer());
    }

    public long getViewHandle() {
      if (viewHandle == 0) {
        System.out.println("kuhu columnHandle =" + columnHandle);
        viewHandle = getNativeColumnView(columnHandle);
      }
      return viewHandle;
    }

    public long getNativeRowCount() {
      return BaseColumnVector.getNativeRowCount(getViewHandle());
    }

    public long getNativeNullCount() {
      if (viewHandle != 0) {
        return BaseColumnVector.getNativeNullCount(getViewHandle());
      }
      return getNativeNullCountColumn(columnHandle);
    }

    private void setNativeNullCount(int nullCount) throws CudfException {
      assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
      assert columnHandle != 0;
      setNativeNullCountColumn(columnHandle, nullCount);
    }

    private DeviceMemoryBufferView getNativeValidPointer() {
      long arg1 = getViewHandle();
      System.out.println("getNativeValidPointer getViewHandle()" + arg1);
      long[] values = BaseColumnVector.getNativeValidPointer(arg1);
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    private DeviceMemoryBufferView getNativeDataPointer() {
      long[] values = BaseColumnVector.getNativeDataPointer(getViewHandle());
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    private DeviceMemoryBufferView getNativeOffsetsPointer() {
      long[] values = BaseColumnVector.getNativeOffsetsPointer(getViewHandle());
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    public DType getNativeType() {
      return DType.fromNative(getNativeTypeId(getViewHandle()));
    }

    public BaseDeviceMemoryBuffer getData() {
      return data;
    }

    public BaseDeviceMemoryBuffer getValid() {
      return valid;
    }

    public BaseDeviceMemoryBuffer getOffsets() {
      return offsets;
    }

    public ArrayList<OffHeapState> getChildrenPointers() {
      long[] values = BaseColumnVector.getChildrenColumnPointers(getColumnHandle());
      ArrayList<OffHeapState> cvs = new ArrayList<>();
      for(int i =0;i < values.length;i++) {
        if (values[i] !=0) {
          cvs.add(new OffHeapState(values[i]));
        }
//        System.out.println(i+"TYPE: "+ DType.fromNative(getNativeTypeId((values[i]))));
      }
      return cvs;
    }

    @Override
    public void noWarnLeakExpected() {
      super.noWarnLeakExpected();
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if(offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }

    @Override
    public String toString() {
      return "(ID: " + id + " " + Long.toHexString(columnHandle == 0 ? viewHandle : columnHandle) + ")";
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long address = 0;

      // Always mark the resource as freed even if an exception is thrown.
      // We cannot know how far it progressed before the exception, and
      // therefore it is unsafe to retry.
      Throwable toThrow = null;
      if (viewHandle != 0) {
        address = viewHandle;
        try {
          deleteColumnView(viewHandle);
        } catch (Throwable t) {
          toThrow = t;
        } finally {
          viewHandle = 0;
        }
        neededCleanup = true;
      }
      if (columnHandle != 0) {
        if (address != 0) {
          address = columnHandle;
        }
        try {
          deleteCudfColumn(columnHandle);
        } catch (Throwable t) {
          if (toThrow != null) {
            toThrow.addSuppressed(t);
          } else {
            toThrow = t;
          }
        } finally {
          columnHandle = 0;
        }
        neededCleanup = true;
      }
      if (data != null || valid != null || offsets != null) {
        try {
          ColumnVector.closeBuffers(data, valid, offsets);
        } catch (Throwable t) {
          if (toThrow != null) {
            toThrow.addSuppressed(t);
          } else {
            toThrow = t;
          }
        } finally {
          data = null;
          valid = null;
          offsets = null;
        }
        neededCleanup = true;
      }
      if (toThrow != null) {
        throw new RuntimeException(toThrow);
      }
      if (neededCleanup) {
        if (logErrorIfNotClean) {
          BaseColumnVector.log.error("A DEVICE COLUMN VECTOR WAS LEAKED (ID: " + id + " " + Long.toHexString(address)+ ")");
          logRefCountDebug("Leaked vector");
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return viewHandle == 0 && columnHandle == 0 && data == null && valid == null && offsets == null;
    }

    /**
     * This returns total memory allocated in device for the ColumnVector.
     * @return number of device bytes allocated for this column
     */
    public long getDeviceMemorySize() {
      long size = valid != null ? valid.getLength() : 0;
      size += offsets != null ? offsets.getLength() : 0;
      size += data != null ? data.getLength() : 0;
      return size;
    }
  }

  ////////
  // Native cudf::column_view life cycle and metadata access methods. Life cycle methods
  // should typically only be called from the OffHeap inner class.
  ////////

  static native int getNativeTypeId(long viewHandle) throws CudfException;

  private static native int getNativeRowCount(long viewHandle) throws CudfException;

  private static native int getNativeNullCount(long viewHandle) throws CudfException;

  private static native void deleteColumnView(long viewHandle) throws CudfException;

  private static native long[] getNativeDataPointer(long viewHandle) throws CudfException;

  static native long[] getNativeOffsetsPointer(long viewHandle) throws CudfException;

  private static native long[] getNativeValidPointer(long viewHandle) throws CudfException;

  protected static native long getChildColumnView(long viewHandle) throws CudfException;

  private static native long makeCudfColumnView(int type, long data, long dataSize, long offsets, long valid, int nullCount, int size, long childLcv);

  private static native long[] getChildrenPointers(long viewHandle) throws CudfException;
  private static native long[] getChildrenColumnPointers(long colHandle) throws CudfException;
  ////////
  // Native methods specific to cudf::column. These either take or create a cudf::column
  // instead of a cudf::column_view so they need to be used with caution. These should
  // only be called from the OffHeap inner class.
  ////////

  /**
   * Delete the column. This is not private because there are a few cases where Table
   * may get back an array of columns pointers and we want to have best effort in cleaning them up
   * on any failure.
   */
  static native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  private static native int getNativeNullCountColumn(long cudfColumnHandle) throws CudfException;

  private static native void setNativeNullCountColumn(long cudfColumnHandle, int nullCount) throws CudfException;

  /**
   * Create a cudf::column_view from a cudf::column.
   * @param cudfColumnHandle the pointer to the cudf::column
   * @return a pointer to a cudf::column_view
   * @throws CudfException on any error
   */
  private static native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  private static native long makeEmptyCudfColumn(int type);

  /**
   * Used for string strip function.
   * Indicates characters to be stripped from the beginning, end, or both of each string.
   */
  protected enum StripType {
    LEFT(0),   // strip characters from the beginning of the string
    RIGHT(1),  // strip characters from the end of the string
    BOTH(2);   // strip characters from the beginning and end of the string
    final int nativeId;

    StripType(int nativeId) { this.nativeId = nativeId; }
  }
}
