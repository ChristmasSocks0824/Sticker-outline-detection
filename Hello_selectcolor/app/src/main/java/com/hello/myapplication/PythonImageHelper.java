package com.hello.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

public class PythonImageHelper {
    private final Python py;

    public PythonImageHelper() {
        this.py = Python.getInstance();

    }

    /**
     * 呼叫 Python OpenCV 進行 Canny 邊緣檢測
     * @param inputImageBytes 輸入圖片的 byte 陣列
     * @param listener 處理結果的回調介面
     */
    public void processCanny(byte[] inputImageBytes, OnProcessListener listener) {
        processInternal("canny_from_image_bytes", inputImageBytes, listener);
    }

    public void processIdentity(byte[] inputImageBytes, OnProcessListener listener) {
        processInternal("identity_from_image_bytes", inputImageBytes, listener);
    }

    private void processInternal(String methodName, byte[] inputImageBytes, OnProcessListener listener) {
        new Thread(() -> {
            try {
                // 呼叫 opencv_process.py
                PyObject module = py.getModule("opencv_process");

                PyObject result = module.callAttr(methodName, inputImageBytes);
                //methodName是字串，可以改變呼叫pyhton中"canny_from_image_bytes"
                //將 Python 回傳的 PyObject 轉換回 Java 的 byte陣列
                byte[] outPng = result.toJava(byte[].class);
                Bitmap outBitmap = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                
                if (listener != null) {
                    listener.onSuccess(outBitmap);
                }
            } catch (Exception e) {
                e.printStackTrace();
                if (listener != null) {
                    listener.onError(e.getMessage());
                }
            }
        }).start();
    }

    // 定義回調介面
    public interface OnProcessListener {
        void onSuccess(Bitmap bitmap);
        void onError(String errorMessage);
    }
}
