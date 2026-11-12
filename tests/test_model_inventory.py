from pm25_project.model_inventory import INVENTORY, inventory_table

def test_pure_cnn_lstm_inventory():
    names = {row[0] for row in INVENTORY}
    assert names == {"CNN-LSTM"}
    table = inventory_table()
    assert table.implementation_status.notna().all()
    required = table.loc[table.assignment_scope.eq("required_primary"), "model_id"].tolist()
    assert required == ["CNN-LSTM"]

