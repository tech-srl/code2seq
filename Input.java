public static boolean f(Bitmap bmp, File file) {
  try {
    BufferedOutputStream bos = new BufferedOutputStream(
      new FileOutputStream(file));
    bmp.compress(Bitmap.CompressFormat.PNG, 90, bos);
    bos.close();
    return true;
  } catch (IOException e) {
    Log.e(TAG, "failed to save frame", e);
  }
  return false;
}