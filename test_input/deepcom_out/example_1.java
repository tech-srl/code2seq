class A {
@Override public void visitLocalVariable(String name,String desc,String signature,Label start,Label end,int index){
  if (index < 1 || currentIndex >= numberOfParameters || name == null)   return;
  parameterNames[currentIndex]=name;
  currentIndex++;
}
}
