package ai.rapids.cudf;

import java.io.Closeable;
import java.io.IOException;

public class TablesAndRows implements Closeable {
    int numRows;
    Table table;
    public TablesAndRows(int numRows, Table table) {
        this.numRows = numRows;
        this.table = table;
    }

    @Override
    public void close() throws IOException {
        if (table != null) {
            table.close();
        }
    }
}