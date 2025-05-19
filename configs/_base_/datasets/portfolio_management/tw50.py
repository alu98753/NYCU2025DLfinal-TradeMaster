
data = dict(
    type = "PortfolioManagementDataset",
    data_path = "data/portfolio_management/tw50",
    train_path = "data/portfolio_management/tw50/train.csv",
    valid_path = "data/portfolio_management/tw50/valid.csv",
    test_path = "data/portfolio_management/tw50/test.csv",
    tech_indicator_list = [
        "high",
        "low",
        "open",
        "close",
        "adjcp",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    length_day = 50,
    initial_amount = 100000,
    transaction_cost_pct = 0.001,
    time_steps = 50
)