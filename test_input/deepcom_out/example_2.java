class A {
public long writeLongGolomb(final long x,final long b) throws IOException {
  return writeLongGolomb(x,b,Fast.mostSignificantBit(b));
}
}
