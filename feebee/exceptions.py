class feebeeError(Exception):
    pass


class NoRowToInsert(feebeeError):
    "Where there's no row to write to a database"
    pass


class NoRowToWrite(feebeeError):
    "When there's no row to write to a CSV file"
    pass


class InvalidGroup(feebeeError):
    pass


class UnknownConfig(feebeeError):
    pass


class ReservedKeyword(feebeeError):
    pass


class InvalidColumns(feebeeError):
    pass


class TableDuplication(feebeeError):
    pass


class NoSuchTableFound(feebeeError):
    pass


class SkipThisTurn(feebeeError):
    pass
