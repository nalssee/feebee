class MapOnError(Exception):
    pass


class NoRowToInsert(MapOnError):
    "Where there's no row to write to a database"
    pass


class NoRowToWrite(MapOnError):
    "When there's no row to write to a CSV file"
    pass


class InvalidGroup(MapOnError):
    pass


class UnknownConfig(MapOnError):
    pass


class ReservedKeyword(MapOnError):
    pass


class InvalidColumns(MapOnError):
    pass


class TableDuplication(MapOnError):
    pass


class NoSuchTableFound(MapOnError):
    pass


class SkipThisTurn(MapOnError):
    pass
