class A {
public static String signatureAlgorithm(String algorithm){
  int index=algorithm.indexOf('_');
  if (index == -1) {
    return algorithm;
  }
  return algorithm.substring(index + 1,algorithm.length());
}
}
