class SequenceGenerator:
  _instance = None
  count = 0

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(SequenceGenerator, cls).__new__(cls)
      cls.count = 0
    return cls._instance

  def get() -> int:
    SequenceGenerator.count += 1
    return SequenceGenerator.count
