

class MJPEG{
  Socket s;
  JpegProvider jpegP;

  public void handleConnection(Socket socket, JpegProvider jpegProvider) throws Exception {
    byte[] data = jpegProvider.getJpeg();
    OutputStream outputStream = socket.getOutputStream();
    outputStream.write((
        "HTTP/1.0 200 OK\r\n" +
        "Server: YourServerName\r\n" +
        "Connection: close\r\n" +
        "Max-Age: 0\r\n" +
        "Expires: 0\r\n" +
        "Cache-Control: no-cache, private\r\n" + 
        "Pragma: no-cache\r\n" + 
        "Content-Type: multipart/x-mixed-replace; " +
        "boundary=--BoundaryString\r\n\r\n").getBytes());
    while (true) {
      data = jpegProvider.getJpeg();
      outputStream.write((
          "--BoundaryString\r\n" +
          "Content-type: image/jpg\r\n". +
          "Content-Length: " +
          data.length +
          "\r\n\r\n").getBytes());
      outputStream.write(data);
      outputStream.write("\r\n\r\n".getBytes());
      outputStream.flush();
    }
  }
}