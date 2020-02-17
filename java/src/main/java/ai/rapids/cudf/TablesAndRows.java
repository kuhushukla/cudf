package ai.rapids.cudf;

public class TablesAndRows {
    int numRows;
    Table table;
    public TablesAndRows(int numRows, Table table) {
        this.numRows = numRows;
        this.table = table;
    }
}