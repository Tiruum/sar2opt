class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        """
        Инициализация.
        - patience (int): Сколько эпох ждать улучшения.
        - verbose (bool): Печатать сообщения о ранней остановке.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Проверить улучшение.
        - val_loss (float): Значение валидационной потери.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                self.early_stop = True
