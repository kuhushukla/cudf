/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.xml.crypto.Data;
import java.io.*;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Similar to a ColumnVector, but the data is stored in host memory and accessible directly from
 * the JVM. This class holds references to off heap memory and is reference counted to know when
 * to release it.  Call close to decrement the reference count when you are done with the column,
 * and call incRefCount to increment the reference count.
 */
public final class HostColumnVector extends BaseHostColumnVector implements AutoCloseable {
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.sizeInBytes;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  @Override
  protected BaseHostColumnVector getChild() {
    return lcv;
  }
  private ListHostColumnVector lcv = null;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;

  /**
   * Create a new column vector with data populated on the host.
   */
  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer, ListHostColumnVector lcv) {
    this(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, null, lcv);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING
   *                           this is the string data stored as bytes.
   * @param hostValidityBuffer Arrow-like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It should be (rows + 1) ints.
   */
  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer, ListHostColumnVector lcv) {
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type != DType.STRING && type != DType.LIST) {
      assert offsetBuffer == null : "offsets are only supported for LIST";
    }
    offHeap = new BaseHostColumnVector.OffHeapState(hostDataBuffer, hostValidityBuffer, offsetBuffer);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    this.lcv = lcv;

    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    offHeap.noWarnLeakExpected();
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
   */
  @Override
  public synchronized void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
  }

  @Override
  public String toString() {
    return "HostColumnVector{" +
        "rows=" + rows +
        ", type=" + type +
        ", nullCount=" + nullCount +
        ", offHeap=" + offHeap +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public HostColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private synchronized HostColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the amount of host memory used to store column/validity data (not metadata).
   */
  public long getHostMemorySize() {
    return offHeap.getHostMemorySize();
  }

  /**
   * Returns the type of this vector.
   */
  public DType getType() {
    return type;
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    if (!nullCount.isPresent()) {
      throw new IllegalStateException("Calculating an unknown null count on the host is not currently supported");
    }
    return nullCount.get();
  }

  /**
   * Returns this column's current refcount
   */
  synchronized int getRefCount() {
    return refCount;
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  public boolean hasValidityVector() {
    return (offHeap.valid != null);
  }

  /**
   * Returns if the vector has nulls.  Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Copy the data to the device.
   */
  public ColumnVector copyToDevice() {
//    System.out.println("KUHU copyToDevice type=" + type + "rows=" + rows + " type.sizeInBytes="+type.sizeInBytes
//    + " this.offHeap.data.length="+ this.offHeap.data.length + "this.offHeap.offset.length"+this.offHeap.offsets.getLength());
    if (rows == 0) {
      return new ColumnVector(type, 0, Optional.of(0L), null, null, null);
    }
    // The simplest way is just to copy the buffers and pass them down.
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer valid = null;
    DeviceMemoryBuffer offsets = null;
    try {
      HostMemoryBuffer hdata = this.offHeap.data;
      if (hdata != null) {
        long dataLen = rows * DType.INT32.sizeInBytes;
        if (type == DType.STRING) {
          // This needs a different type
          dataLen = getEndStringOffset(rows - 1);
          if (dataLen == 0 && getNullCount() == 0) {
            // This is a work around to an issue where a column of all empty strings must have at
            // least one byte or it will no t be interpreted correctly.
            dataLen = 1;
          }
        }
        if (type == DType.LIST) {
          // This needs a different type
          dataLen = this.offHeap.data.length;
          if (dataLen == 0 && getNullCount() == 0) {
            // This is a work around to an issue where a column of all empty strings must have at
            // least one byte or it will not be interpreted correctly.
            dataLen = 1;
          }
        }
        data = DeviceMemoryBuffer.allocate(dataLen);
        System.out.println("KUHU data copytodevice len=" + data.getLength() + " dataLen" + dataLen);
        data.copyFromHostBuffer(hdata, 0, dataLen);
      }
      HostMemoryBuffer hvalid = this.offHeap.valid;
      if (hvalid != null) {
        long validLen = ColumnVector.getNativeValidPointerSize((int)rows);
        valid = DeviceMemoryBuffer.allocate(validLen);
        valid.copyFromHostBuffer(hvalid, 0 , validLen);
      }

      HostMemoryBuffer hoff = this.offHeap.offsets;
      byte[] tmp = new byte[(int)this.offHeap.offsets.length];
      System.out.println("KUHU TMP OFFSETS c2d==========");
      this.offHeap.offsets.getBytes(tmp,0,0,tmp.length);
      for (int i = 0; i < tmp.length; i++) {
        System.out.print((tmp[i]) + " ");
      }
      byte[] tmpD = new byte[(int)this.offHeap.data.length];
      System.out.println("KUHU TMP DATA ==========");
      this.offHeap.data.getBytes(tmpD,0,0,tmpD.length);
      for (int i = 0; i < tmpD.length; i++) {
        System.out.print((tmpD[i]) + " ");
      }
      if (hoff != null) {
        long offsetsLen = OFFSET_SIZE * (rows + 1);
        System.out.println("KUHU offsetsLen=" +offsetsLen);
        offsets = DeviceMemoryBuffer.allocate(offsetsLen);
        offsets.copyFromHostBuffer(hoff, 0 , offsetsLen);
      }
      ColumnVector ret = null;
      if (this.type != DType.LIST) {
        ret = new ColumnVector(type, rows, nullCount, data, valid, offsets);
      } else {
        ListColumnVector listColumnVector = makeLcv(this.lcv);
        ret = new ColumnVector(type, rows, nullCount, data, valid, offsets, listColumnVector);
      }
      data = null;
      valid = null;
      offsets = null;
      System.out.println("KUHU C2D ret native view =" +ret.getNativeView()  + " ret.data=" + ret.offHeap.getData().length);
      return ret;
    } finally {
      if (data != null) {
        data.close();
      }
      if (valid != null) {
        valid.close();
      }
      if (offsets != null) {
        offsets.close();
      }
    }
  }

  ListColumnVector makeLcv(ListHostColumnVector lhcv) {
    DeviceMemoryBuffer tmpValid = null;
    DeviceMemoryBuffer tmpOffsets = null;
    if (lhcv.offHeap.valid != null) {
      tmpValid=DeviceMemoryBuffer.allocate(lhcv.offHeap.valid.length);
      tmpValid.copyFromHostBuffer(lhcv.offHeap.valid, 0, lhcv.offHeap.valid.length);
    }
    if (lhcv.offHeap.offsets != null) {
      tmpOffsets=DeviceMemoryBuffer.allocate(lhcv.offHeap.offsets.length);
      tmpOffsets.copyFromHostBuffer(lhcv.offHeap.offsets, 0, lhcv.offHeap.offsets.length);
    }

    if (lhcv.childLcv == null) {
      DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(this.offHeap.data.length);
      data.copyFromHostBuffer(this.offHeap.data, 0, this.offHeap.data.length);
      return new ListColumnVector(lhcv.type, (int)lhcv.rows, data, tmpValid, tmpOffsets, null);
    }
    ListColumnVector listColumnVector = new ListColumnVector(lhcv.type, (int)lhcv.rows, null, tmpValid, tmpOffsets, makeLcv(lhcv.childLcv));

    return listColumnVector;
  }
  /////////////////////////////////////////////////////////////////////////////
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      return BitVectorHelper.isNull(offHeap.valid, index);
    }
    return false;
  }

  /**
   * For testing only.  Allows null checks to go past the number of rows, but not past the end
   * of the buffer.  NOTE: If the validity vector was allocated by cudf itself it is not
   * guaranteed to have the same padding, but for all practical purposes it does.  This is
   * just to verify that the buffer was allocated and initialized properly.
   */
  boolean isNullExtendedRange(long index) {
    long maxNullRow = BitVectorHelper.getValidityAllocationSizeInBytes(rows) * 8;
    assert (index >= 0 && index < maxNullRow) : "TEST: index is out of range 0 <= " + index + " <" +
        " " + maxNullRow;
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      return BitVectorHelper.isNull(offHeap.valid, index);
    }
    return false;
  }

  /**
   * Get access to the raw host buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the host for this
   * to work.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public HostMemoryBuffer getHostBufferFor(BufferType type) {
    HostMemoryBuffer srcBuffer = null;
    switch(type) {
      case VALIDITY:
        srcBuffer = offHeap.valid;
        break;
      case OFFSET:
        srcBuffer = offHeap.offsets;
        break;
      case DATA:
        srcBuffer = offHeap.data;
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");
    }
    return srcBuffer;
  }

  void copyHostBufferBytes(byte[] dst, int dstOffset, BufferType src, long srcOffset,
                           int length) {
    assert dstOffset >= 0;
    assert srcOffset >= 0;
    assert length >= 0;
    assert dstOffset + length <= dst.length;

    HostMemoryBuffer srcBuffer = getHostBufferFor(src);

    assert srcOffset + length <= srcBuffer.length : "would copy off end of buffer "
        + srcOffset + " + " + length + " > " + srcBuffer.length;
    UnsafeMemoryAccessor.getBytes(dst, dstOffset,
        srcBuffer.getAddress() + srcOffset, length);
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public byte getByte(long index) {
    assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.data.getByte(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type == DType.INT16 || type == DType.UINT16;
    assertsForGet(index);
    return offHeap.data.getShort(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type == DType.INT32 || type == DType.UINT32 || type == DType.TIMESTAMP_DAYS;
    assertsForGet(index);
    return offHeap.data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the starting byte offset for the string at index
   */
  long getStartStringOffset(long index) {
    assert type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    return offHeap.offsets.getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   */
  long getEndStringOffset(long index) {
    assert type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    // The offsets has one more entry than there are rows.
    return offHeap.offsets.getInt((index + 1) * 4);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    // Timestamps with time values are stored as longs
    assert type == DType.INT64 || type == DType.UINT64 || type.hasTimeResolution();
    assertsForGet(index);
    return offHeap.data.getLong(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type == DType.FLOAT32;
    assertsForGet(index);
    return offHeap.data.getFloat(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type == DType.FLOAT64;
    assertsForGet(index);
    return offHeap.data.getDouble(index * type.sizeInBytes);
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.data.getBoolean(index * type.sizeInBytes);
  }

  /**
   * Get the raw UTF8 bytes at index.  This API is faster than getJavaString, but still not
   * ideal because it is copying the data onto the heap.
   */
  public byte[] getUTF8(long index) {
    assert type == DType.STRING;
    assertsForGet(index);
    int start = offHeap.offsets.getInt(index * OFFSET_SIZE);
    int size = offHeap.offsets.getInt((index + 1) * OFFSET_SIZE) - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.data.getBytes(rawData, 0, start, size);
    }
    return rawData;
  }

  public List getListParent(long rowIndex) throws Exception {
    return getListMain(rowIndex, this);
  }

  public List getListMain(long rowIndex, BaseHostColumnVector mainCv) throws Exception {
    System.out.println("KUHU child.type="+mainCv + " mainCv.childLcv" +mainCv.getChild());

    if (mainCv.getChild().type == DType.LIST) {
      List retList = new ArrayList();
      byte[] offsetBytes = new byte[(int)mainCv.offHeap.offsets.length];
      mainCv.offHeap.offsets.getBytes(offsetBytes, 0, 0, mainCv.offHeap.offsets.length);
      System.out.println("maincv offsets");
      for (int i =0; i< offsetBytes.length;i++) {
        System.out.print((offsetBytes[i]) + " ");
      }
      int start = mainCv.offHeap.offsets.getInt(rowIndex*DType.INT32.getSizeInBytes());
      int end = mainCv.offHeap.offsets.getInt((rowIndex+1)*DType.INT32.getSizeInBytes());
      if (start == end) {
        retList.add(getListMain(start, mainCv.getChild()));
      }
      for(int i =start;i<end;i++) {
        retList.add(getListMain(i, mainCv.getChild()));
      }
      return retList;
    } else {
      int start = mainCv.offHeap.offsets.getInt(rowIndex*DType.INT32.getSizeInBytes());
      int end = mainCv.offHeap.offsets.getInt((rowIndex+1)*DType.INT32.getSizeInBytes());
      System.out.println("KUHU getListMain start ="+start + " end="+end + "this.lcv.type="+this.lcv.type);
      byte[] tmpD = new byte[(int)this.offHeap.data.length];
      System.out.println("KUHU getlist DATA ==========");
      this.offHeap.data.getBytes(tmpD,0,0,tmpD.length);

      int size = (end-start)*mainCv.getChild().type.getSizeInBytes();
      byte[] rawData = new byte[size];
      if (size > 0) {
        offHeap.data.getBytes(rawData, 0, start*mainCv.getChild().type.getSizeInBytes(), size);
      }

      System.out.println("KUHU rawdata========"+rawData.length);
      for (int i =0; i < rawData.length;i++) {
        System.out.print((rawData[i]) + " ");
      }
      ByteArrayInputStream bais = new ByteArrayInputStream(rawData);
      DataInputStream dataInputStream = new DataInputStream(bais);
      List<Integer> list = new ArrayList<>();
      System.out.println("KUHU ELEMENTS");
      while (dataInputStream.available() > 0) {
        list.add(dataInputStream.readInt());
//        System.out.println(list.get(list.size()-1));
      }
      System.out.println("retlist="+ list);
      return list;
    }
  }
  public List getList(long index) throws IOException, ClassNotFoundException {
    System.out.println("KUHU type = " + type);
    //assert type == DType.LIST;
    HostMemoryBuffer offsets = lcv.offHeap.offsets;
    HostMemoryBuffer data = offHeap.data;
    int dataLen = (int)data.length;
    System.out.println("KUHU data = " + data.getLength());
    System.out.println("KUHU offsets.len = " + offsets.length);
//    System.out.println("KUHU index * OFFSET_SIZE = "+index * OFFSET_SIZE);
    int start = offsets.getInt(index*DType.INT32.getSizeInBytes())*DType.INT32.getSizeInBytes();
    System.out.println("KUHU start = " + start);
//    long endIndex = (offsets.length / OFFSET_SIZE - 1) * OFFSET_SIZE;

    long dataStart = data.address + start;
    int end = offsets.getInt((index+1)*DType.INT32.getSizeInBytes())*DType.INT32.getSizeInBytes();
    byte[] offsetBytes = new byte[(int)offsets.length];
    offsets.getBytes(offsetBytes, 0, 0, offsets.length);
    byte[] dataBytes = new byte[(int)data.length];
    data.getBytes(dataBytes,0,0,dataBytes.length);
    System.out.println("KUHU ALL DATA========"+dataBytes.length);
    for (int i =0; i< dataBytes.length;i++) {
      System.out.print((dataBytes[i]) + " ");
    }
    System.out.println("KUHU offsetBytes========"+offsetBytes.length);
    System.out.println("KUHU ListColumnVector offsets =" + this.lcv.offHeap.offsets.address + " len =" +
        this.lcv.offHeap.offsets.length);
    for (int i =0; i< offsetBytes.length;i++) {
      System.out.print((offsetBytes[i]) + " ");
    }
    int size = (int) (end - start);
    byte[] rawData = new byte[dataLen];
    System.out.println("KUHU end = " + end + " start=" + start + "dataStart=" + dataStart + " size=" + size);
    if (size > 0) {
      offHeap.data.getBytes(rawData, 0, start, size);
    }

    System.out.println("KUHU rawdata========"+rawData.length);
    for (int i =0; i < rawData.length;i++) {
      System.out.print((rawData[i]) + " ");
    }
    ByteArrayInputStream bais = new ByteArrayInputStream(rawData);
    DataInputStream dataInputStream = new DataInputStream(bais);
    List<Integer> list = new ArrayList<>();
      System.out.println("KUHU ELEMENTS");
      while (dataInputStream.available() > 0) {
        list.add(dataInputStream.readInt());
//        System.out.println(list.get(list.size()-1));
      }
    return list;
  }
  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    byte[] rawData = getUTF8(index);
    return new String(rawData, StandardCharsets.UTF_8);
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows) {
    return new Builder(type, rows, 0);
  }

  public static Builder builder(DType type, DType baseType, int rows, int bufferLen) {
    return new Builder(type, baseType, rows, bufferLen);
  }
  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string or list data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param rows the number of rows this builder can hold
   * @param bufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows, long bufferSize) {
    return new HostColumnVector.Builder(type, DType.INT32, rows, bufferSize);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static HostColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static HostColumnVector build(int rows, long stringBufferSize, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(DType.STRING, rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static HostColumnVector build(DType listType, int rows, long stringBufferSize, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(listType, rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedBytes(byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedShorts(short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedInts(int... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedLongs(long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static HostColumnVector fromStrings(String... values) {
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (String s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    if (nullCount > 0) {
      return build(rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
  }

  public static<T> HostColumnVector fromLists(DType type, int levels, List<T>... values) {
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (List s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.size()*type.getSizeInBytes();
      }
    }
    //ADD support for nulls
//    if (nullCount > 0) {
//      return build(rows, bufferSize, (b) -> b.appendBoxed(values));
//    }
    return build(DType.LIST, rows, bufferSize, (b) -> {
      for (List s: values) {
        System.out.println("KUHU from lists" + s);
        b.appendList(DType.LIST, type, 0, values.length + 1 , s);
      }
      for(HostMemoryBuffer myOffsets: b.offsets) {
        byte[] tmp = new byte[(int)myOffsets.length];
        System.out.println("KUHU TMP OFFSETS==========");
        myOffsets.getBytes(tmp,0,0,tmp.length);
        for (int i = 0; i < tmp.length; i++) {
          System.out.print((tmp[i]) + " ");
        }
      }
    });
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBytes(Byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedBytes(Byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedShorts(Short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedShorts(Short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedInts(Integer... values) {
    return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedInts(Integer... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedLongs(Long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedLongs(Long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Build
   */
  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final DType type;
    private DType baseType;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private ArrayList<HostMemoryBuffer> offsets = new ArrayList<>();
    private ArrayList<Integer> currentOffsets = new ArrayList<>();
    private long currentIndex = 0;
    private long nullCount;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    //change here
    Builder(DType type, long rows, long stringBufferSize) {
      this.type = type;
      this.rows = rows;
      if (type == DType.STRING) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets.add(HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE));
        // The first offset is always 0
        this.offsets.get(0).setInt(0, 0);
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
      }
    }

    //change here
    Builder(DType type, DType baseType, long rows, long stringBufferSize) {
      this.type = type;
      this.baseType = baseType;
      this.rows = rows;
      if (type == DType.LIST) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
//        this.offsets.add(HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE));
        // The first offset is always 0
//        this.offsets.get(0).setInt(0, 0);
      } else {
        this.data = HostMemoryBuffer.allocate(rows * baseType.sizeInBytes);
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(DType type, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
      this.type = type;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      data.setMemory(currentIndex * type.sizeInBytes, count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type == DType.INT16 || type == DType.UINT16;
      assert currentIndex < rows;
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert (type == DType.INT32 || type == DType.UINT32 || type == DType.TIMESTAMP_DAYS);
      assert currentIndex < rows;
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type == DType.INT64 || type == DType.UINT64 || type == DType.TIMESTAMP_MILLISECONDS ||
          type == DType.TIMESTAMP_MICROSECONDS || type == DType.TIMESTAMP_NANOSECONDS ||
          type == DType.TIMESTAMP_SECONDS;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type == DType.FLOAT32;
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type == DType.FLOAT64;
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public Builder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public Builder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public Builder appendUTF8String(byte[] value, int offset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert offset >= 0;
      assert length >= 0;
      assert value.length + offset <= length;
      assert type == DType.STRING;
      assert currentIndex < rows;
      // just for strings we want to throw a real exception if we would overrun the buffer
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentStringByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, offset, length);
      }
      currentStringByteIndex += length;
      currentIndex++;
      offsets.get(0).setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      return this;
    }

    public Builder appendList(DType type, DType baseType, int level, int prevSize, List list) {
      if(list.get(0) instanceof List) {
        int newSize = 0;
        System.out.println("KUHU LSIST OF LISTS");
        System.out.println("KUHU offsets.size()=" + offsets.size());
        List<List> tmpList = list;
        for(List insideList: tmpList) {
          newSize ++;
        }
        if (offsets.size() <= level) {
          System.out.println("1 KUHU offsets.size()=" + offsets.size());
          System.out.println("1 KUHU level" + level);
          this.offsets.add(level, HostMemoryBuffer.allocate(prevSize * OFFSET_SIZE));
          this.offsets.get(level).setInt(0,0);
          this.currentOffsets.add(level, OFFSET_SIZE);
        }
        for(List insideList: tmpList) {
          System.out.println("KUHU LSIST OF LISTS insideList=" + insideList + "level=" + level);
          appendList(type, baseType, level + 1, prevSize + newSize, insideList);
        }
//        this.offsets.add(HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE));

        System.out.println("KUHU offsets.size()=" + offsets.size());
        System.out.println("KUHU currOffsets.size()=" + currentOffsets.size());
        System.out.println("KUHU level" + level);
        this.offsets.get(level).setInt(this.currentOffsets.get(level), this.offsets.get(level).getInt(this.currentOffsets.get(level)- OFFSET_SIZE) + list.size());
        this.currentOffsets.set(level, this.currentOffsets.get(level)+OFFSET_SIZE);
        return this;
      } else {
        System.out.println("KUHU MAIN list" + list + " level =" + level + currentStringByteIndex*baseType.getSizeInBytes());
        assert list != null : "appendNull must be used to append null strings";
//      assert offset >= 0;
//      assert length >= 0;
//      assert value.length + offset <= length;
        int length = list.size();
//        assert currentIndex < rows;
        // just for strings we want to throw a real exception if we would overrun the buffer
        long oldLen = data.getLength();
        long newLen = oldLen;
        while (currentStringByteIndex*baseType.getSizeInBytes() + list.size() * baseType.getSizeInBytes() > newLen) {
          newLen *= 2;
        }
        if (newLen > Integer.MAX_VALUE) {
          throw new IllegalStateException("A string buffer is not supported over 2GB in size");
        }
        System.out.println("KUHU append list" + list + " new len=" + newLen + " oldLen=" + oldLen);
        if (newLen != oldLen) {
          // need to grow the size of the buffer.
          HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
          try {
            newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex*baseType.getSizeInBytes());
            data.close();
            data = newData;
            newData = null;
          } finally {
            if (newData != null) {
              newData.close();
            }
          }
        }
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        byte[] listBytes = null;
        try {
          //out = new ObjectOutputStream(bos);
          for (int i = 0; i < list.size(); i++) {
            dos.writeInt((int) list.get(i));
          }
          dos.flush();
          listBytes = bos.toByteArray();
          ByteArrayInputStream bais = new ByteArrayInputStream(listBytes);
          DataInputStream in = new DataInputStream(bais);

          System.out.println("KUHU TMP ELEMENTS");
          System.out.println(in.readInt());
          System.out.println(in.readInt());
          System.out.println(in.readInt());
        } catch (IOException e) {
          // ignore close exception
        }
        if (length > 0) {
          System.out.println("KUHU listBytes========" + listBytes.length);
          for (int i = 0; i < listBytes.length; i++) {
            System.out.print((listBytes[i]) + " ");
          }
          System.out.println("KUHU INDEX=" + currentStringByteIndex * baseType.getSizeInBytes());
          data.setBytes(currentStringByteIndex * baseType.getSizeInBytes(), listBytes, 0, listBytes.length);
          byte[] tmpArr = new byte[(int) data.length];
          data.getBytes(tmpArr, 0, 0, tmpArr.length);
          System.out.println("KUHU tmpArr========" + tmpArr.length);
          for (int i = 0; i < tmpArr.length; i++) {
            System.out.print((tmpArr[i]) + " ");
          }
        }
        //print again
        currentStringByteIndex += length;
        currentIndex++;
        if (offsets.size() <= level) {
          this.offsets.add(level, HostMemoryBuffer.allocate(prevSize * OFFSET_SIZE));
          offsets.get(level).setInt(0, 0);
          this.currentOffsets.add(level, OFFSET_SIZE);
        }
        System.out.println("KUHU currentIndex=" + currentIndex);
        System.out.println("KUHU rows=" + rows);
        System.out.println("KUHU offset size=" + OFFSET_SIZE);
        System.out.println("KUHU currentStringByteIndex=" + currentStringByteIndex);
        System.out.println("KUHU this.currentOffsets.get(level)=" + this.currentOffsets.get(level));
        System.out.println("KUHU this.offsets.get(level) size=" + this.offsets.get(level).length);
//        offsets.get(level).setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
        this.offsets.get(level).setInt(this.currentOffsets.get(level), currentStringByteIndex);
        this.currentOffsets.set(level,this.currentOffsets.get(level)+OFFSET_SIZE);
        //debug
        byte[] offsetBytes = new byte[(int) prevSize * OFFSET_SIZE];
        offsets.get(level).getBytes(offsetBytes, 0, 0, offsetBytes.length);
        System.out.println("Setting KUHU offsetBytes========" + offsetBytes.length);
        for (int i = 0; i < offsetBytes.length; i++) {
          System.out.print((offsetBytes[i]) + " ");
        }
        System.out.println("KUHU final data length=" + data.getLength());
        //debug
      }
      return this;
    }


//    public Builder appendListsOfLists(DType type, DType baseType, List<List<Integer>> list) {
//      //use flatmap later
//      Object[] array = Stream.of(list).flatMap(List::stream).toArray();
//      List<Integer> myOffsets = new ArrayList<Integer>();
//      myOffsets.add(0);
//      int currentOffsetIndex = 0;
//      for (List<Integer> eachList: list) {
//        myOffsets.add(myOffsets.get(currentOffsetIndex) + eachList.size());
//        currentOffsetIndex++;
//      }
//      System.out.println("KUHU myOffset=" + myOffsets);
//      List<Integer> flat = list.stream().flatMap(x -> x.stream())
//          .collect(Collectors.toList());
//      System.out.println("KUHU ARRAY = " + flat.size());
//      //extract based on baseType
//      List<Integer> dataList = new ArrayList<>();
//      for (int i=0;i<flat.size();i++) {
//        dataList.add(flat.get(i));
//      }
//      System.out.println("KUHU DATA LIST = " + dataList.get(0) + " " + dataList.get(1));
//      assert list != null : "appendNull must be used to append null strings";
////      assert offset >= 0;
////      assert length >= 0;
////      assert value.length + offset <= length;
//      int length = flat.size()*baseType.getSizeInBytes();
//      assert currentIndex < rows;
//      // just for strings we want to throw a real exception if we would overrun the buffer
//      long oldLen = data.getLength();
//      long newLen = oldLen;
//      while (currentStringByteIndex + flat.size()*baseType.getSizeInBytes() > newLen) {
//        newLen *= 2;
//      }
//      if (newLen > Integer.MAX_VALUE) {
//        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
//      }
//      System.out.println("KUHU append list" + flat + " new len=" + newLen + " oldLen=" + oldLen);
//      if (newLen != oldLen) {
//        // need to grow the size of the buffer.
//        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
//        try {
//          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
//          data.close();
//          data = newData;
//          newData = null;
//        } finally {
//          if (newData != null) {
//            newData.close();
//          }
//        }
//      }
//      ByteArrayOutputStream bos = new ByteArrayOutputStream();
//      DataOutputStream dos = new DataOutputStream(bos);
//      byte[] listBytes = null;
//      try {
//        //out = new ObjectOutputStream(bos);
//        for (int i = 0;i< flat.size();i++) {
//          dos.writeInt((int)(flat.get(i)));
//        }
//        dos.flush();
//        listBytes = bos.toByteArray();
//        System.out.println("KUHU listBytes========"+listBytes.length);
//        for (int i =0; i< listBytes.length;i++) {
//          System.out.print((listBytes[i]) + " ");
//        }
//        ByteArrayInputStream bais = new ByteArrayInputStream(listBytes);
//        DataInputStream in = new DataInputStream(bais);
//
//        System.out.println("KUHU TMP ELEMENTS");
//        System.out.println(in.readInt());
//        System.out.println(in.readInt());
//        System.out.println(in.readInt());
//      }  catch (IOException e) {
//        // ignore close exception
//      }
//      if (length > 0) {
//        data.setBytes(currentStringByteIndex, listBytes, 0, length);
//      }
//      //print again
//      currentStringByteIndex += length;
//      currentIndex++;
//      if (offsets.size() == 0) {
//        this.offsets.add(HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE));
//      }
//      offsets.get(0).setInt(0, 0);
//      System.out.println("KUHU currentIndex="+currentIndex);
//      System.out.println("KUHU offset size="+OFFSET_SIZE);
//      System.out.println("KUHU currentStringByteIndex="+currentStringByteIndex);
//      offsets.get(0).setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
//      //debug
//      byte[] offsetBytes = new byte[listBytes.length];
//      offsets.get(0).getBytes(offsetBytes, 0, 0, listBytes.length);
//      System.out.println("Setting KUHU offsetBytes========"+offsetBytes.length);
//      for (int i =0; i< offsetBytes.length;i++) {
//        System.out.print((offsetBytes[i]) + " ");
//      }
//      System.out.println("KUHU final data length=" + data.getLength());
//      //debug
//      return this;
//    }

    public Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type == DType.INT16 || type == DType.UINT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert (type == DType.INT32 || type == DType.UINT32 || type == DType.TIMESTAMP_DAYS);
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type == DType.INT64 || type == DType.UINT64 || type == DType.TIMESTAMP_MILLISECONDS ||
          type == DType.TIMESTAMP_MICROSECONDS || type == DType.TIMESTAMP_NANOSECONDS ||
          type == DType.TIMESTAMP_SECONDS;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type == DType.FLOAT32;
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
      assert type == DType.FLOAT64;
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Byte... values) throws IndexOutOfBoundsException {
      for (Byte b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Boolean... values) throws IndexOutOfBoundsException {
      for (Boolean b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b ? (byte) 1 : (byte) 0);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Short... values) throws IndexOutOfBoundsException {
      for (Short b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Integer... values) throws IndexOutOfBoundsException {
      for (Integer b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Long... values) throws IndexOutOfBoundsException {
      for (Long b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Float... values) throws IndexOutOfBoundsException {
      for (Float b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Double... values) throws IndexOutOfBoundsException {
      for (Double b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(String... values) throws IndexOutOfBoundsException {
      for (String b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }
    // TODO see if we can remove this...
    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The CudfColumn based on this builder values
     */
    public final Builder append(HostColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type == type;

      if (type == DType.STRING) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.data,
            0L,
            columnVector.getRowCount() * type.sizeInBytes);
      }

      //As this is doing the append on the host assume that a null count is available
      long otherNc = columnVector.getNullCount();
      if (otherNc != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intCudfColumn to this
        BitVectorHelper.append(columnVector.offHeap.valid, valid, currentIndex,
            columnVector.rows);
        nullCount += otherNc;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = ColumnVector.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type == DType.STRING) {
        offsets.get(0).setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      }
      return this;
    }

    /**
     * Set a specific index to null.
     * @param index
     */
    public final Builder setNullAt(long index) {
      assert index < rows;

      // add null
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
      nullCount += BitVectorHelper.setNullAt(valid, index);
      return this;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final HostColumnVector build() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      if (type == DType.LIST) {
        ListHostColumnVector lcv = makeLhcv(1);
        HostColumnVector cv = new HostColumnVector(type,
            rows, Optional.of(nullCount), data, valid, offsets.get(0), lcv);
        built = true;
        return cv;
      } else {
        HostColumnVector cv = new HostColumnVector(type,
            rows, Optional.of(nullCount), data, valid, offsets.get(0), null);
        built = true;
        return cv;
      }
    }

    ListHostColumnVector makeLhcv(int level) {
      if (level >= offsets.size()) {
        return new ListHostColumnVector(this.baseType,(int)this.rows, null,this.valid,null);
      }
      ListHostColumnVector listHostColumnVector = new ListHostColumnVector(this.type, (int)this.rows, null,this.valid,this.offsets.get(level));
      listHostColumnVector.childLcv = makeLhcv(level+1);
      return listHostColumnVector;
    }
    /**
     * Build list metadata
     */
    public final ListHostColumnVector buildListMetadata(DType type, HostMemoryBuffer offsets,
                                                        HostMemoryBuffer validity) {
      ListHostColumnVector lcv = new ListHostColumnVector(type,(int)this.rows, null, validity, offsets);
      return lcv;
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      try (HostColumnVector tmp = build()) {
        return tmp.copyToDevice();
      }
    }

    /**
     * Close this builder and free memory if the CudfColumn wasn't generated. Verifies that
     * the data was released even in the case of an error.
     */
    @Override
    public final void close() {
      if (!built) {
        data.close();
        data = null;
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          // close all
          offsets.get(0).close();
          offsets = null;
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      return "Builder{" +
          "data=" + data +
          "type=" + type +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", built=" + built +
          '}';
    }
  }
}
