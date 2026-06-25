package com.hello.myapplication;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.LayerDrawable;
import android.graphics.drawable.InsetDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.github.chrisbanes.photoview.PhotoView;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "CameraApp";
    private static final int CAMERA_PERMISSION_CODE = 100;

    private PreviewView viewFinder;
    private ImageView imgOriginal;
    private ImageView imgResult;
    private TextView txtStatus;

    private View fullscreenContainer;
    private PhotoView imgFullscreen;
    
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    
    private byte[] inputImageBytes;
    private Python py;
    private boolean isCameraActive = false;

    // 模式選擇相關
    private Spinner modeSpinner;
    private String selectedMode = "Object Recognition";

    // Gamma 調整相關
    private TextView txtGammaValue;
    private float currentGamma = 0.7f;

    // Area (Noise Floor) 調整相關
    private TextView txtAreaValue;
    private int currentNoiseFloor = 300;

    // Target Index 相關
    private TextView txtTargetIndex;
    private int selectedTargetIndex = 0;

    // 相機控制與縮放偵測
    private Camera camera;
    private ScaleGestureDetector scaleGestureDetector;

    // HSV 顏色管理相關
    private List<HSVRange> currentHsvRanges = new ArrayList<>();

    public static class HSVRange {
        String name;
        int hLow, sLow, vLow;
        int hHigh, sHigh, vHigh;

        public HSVRange(String name, int hL, int sL, int vL, int hH, int sH, int vH) {
            this.name = name;
            this.hLow = hL; this.sLow = sL; this.vLow = vL;
            this.hHigh = hH; this.sHigh = sH; this.vHigh = vH;
        }

        public int[][] toArray() {
            return new int[][]{{hLow, sLow, vLow}, {hHigh, sHigh, vHigh}};
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFinder = findViewById(R.id.viewFinder);
        imgOriginal = findViewById(R.id.imgOriginal);
        imgResult = findViewById(R.id.imgResult);
        txtStatus = findViewById(R.id.txtStatus);

        fullscreenContainer = findViewById(R.id.fullscreenContainer);
        imgFullscreen = findViewById(R.id.imgFullscreen);

        imgOriginal.setOnClickListener(v -> showFullscreenImage(imgOriginal));
        imgResult.setOnClickListener(v -> showFullscreenImage(imgResult));
        
        imgFullscreen.setOnViewTapListener((view, x, y) -> hideFullscreenImage());
        fullscreenContainer.setOnClickListener(v -> hideFullscreenImage());

        Button btnToggleCamera = findViewById(R.id.btnToggleCamera);
        Button btnCapture = findViewById(R.id.btnCapture);
        Button btnProcessImage = findViewById(R.id.btnProcessImage);
        Button btnZoomIn = findViewById(R.id.btnZoomIn);
        Button btnZoomOut = findViewById(R.id.btnZoomOut);

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        py = Python.getInstance();

        btnToggleCamera.setOnClickListener(v -> toggleCamera());
        btnCapture.setOnClickListener(v -> takePhoto());
        btnProcessImage.setOnClickListener(v -> processImageWithPython());
        btnZoomIn.setOnClickListener(v -> adjustZoom(1.2f));
        btnZoomOut.setOnClickListener(v -> adjustZoom(0.8f));

        setupModeSpinner();

        // 初始化 Gamma
        txtGammaValue = findViewById(R.id.txtGammaValue);
        Button btnGammaUp = findViewById(R.id.btnGammaUp);
        Button btnGammaDown = findViewById(R.id.btnGammaDown);
        btnGammaUp.setOnClickListener(v -> adjustGamma(0.1f));
        btnGammaDown.setOnClickListener(v -> adjustGamma(-0.1f));

        // 初始化 Area (Noise Floor)
        txtAreaValue = findViewById(R.id.txtAreaValue);
        Button btnKeyboardInput = findViewById(R.id.btnKeyboardInput);
        btnKeyboardInput.setOnClickListener(v -> showNoiseFloorInputDialog());

        // 初始化 Target Index
        txtTargetIndex = findViewById(R.id.txtTargetIndex);
        Button btnTargetInput = findViewById(R.id.btnTargetInput);
        btnTargetInput.setOnClickListener(v -> showTargetIndexInputDialog());

        // 初始化 HSV 顏色管理
        initDefaultHsvRanges();
        Button btnManageColors = findViewById(R.id.btnManageColors);
        btnManageColors.setOnClickListener(v -> showColorManagementDialog());

        cameraExecutor = Executors.newSingleThreadExecutor();
        setupZoomGesture();
        checkCameraPermission();
    }

    private void setupModeSpinner() {
        modeSpinner = findViewById(R.id.modeSpinner);
        String[] displayModes = {"mode1(Contour Detection)", "mode2(Object Recognition)", "mode3(Gamma Pre-image)", "mode4(HSV_findContour )", "mode5(LAB_findContour )", "mode6(LAB_Debug)"};
        String[] technicalModes = {"Contour Detection", "Object Recognition", "Debug Pre-processing", "HSV_findContour", "LAB_findContour", "LAB_Debug"};
        
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, displayModes);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner.setAdapter(adapter);
        modeSpinner.setSelection(1); // Default to Object Recognition

        modeSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                selectedMode = technicalModes[position];
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });
    }

    private void adjustGamma(float delta) {
        currentGamma += delta;
        if (currentGamma < 0.1f) currentGamma = 0.1f;
        if (currentGamma > 5.0f) currentGamma = 5.0f;
        txtGammaValue.setText(String.format(java.util.Locale.US, "%.1f", currentGamma));
    }

    private void showNoiseFloorInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("輸入 Area (Noise Floor)");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER);
        input.setText(String.valueOf(currentNoiseFloor));
        builder.setView(input);

        builder.setPositiveButton("確定", (dialog, which) -> {
            String text = input.getText().toString();
            try {
                int value = Integer.parseInt(text);
                if (value >= 100 && value <= 10000) {
                    currentNoiseFloor = value;
                    txtAreaValue.setText(String.valueOf(currentNoiseFloor));
                    Toast.makeText(this, "Area 已更新", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "超出範圍 (100-10000)", Toast.LENGTH_SHORT).show();
                }
            } catch (NumberFormatException e) {
                Toast.makeText(this, "請輸入有效數字", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("取消", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private void showTargetIndexInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("輸入要鎖定的物件編號");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER);
        input.setText(String.valueOf(selectedTargetIndex));
        builder.setView(input);

        builder.setPositiveButton("確定", (dialog, which) -> {
            String text = input.getText().toString();
            try {
                int value = Integer.parseInt(text);
                if (value >= 0 && value <= 50) {
                    selectedTargetIndex = value;
                    txtTargetIndex.setText(String.valueOf(selectedTargetIndex));
                    Toast.makeText(this, "目標編號已設為: " + selectedTargetIndex, Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "請輸入 0-50 之間的編號", Toast.LENGTH_SHORT).show();
                }
            } catch (NumberFormatException e) {
                Toast.makeText(this, "請輸入有效數字", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("取消", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private void setupZoomGesture() {
        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(@NonNull ScaleGestureDetector detector) {
                if (camera == null) return false;
                float currentZoomRatio = camera.getCameraInfo().getZoomState().getValue() != null ? 
                        camera.getCameraInfo().getZoomState().getValue().getZoomRatio() : 1.0f;
                float delta = detector.getScaleFactor();
                camera.getCameraControl().setZoomRatio(currentZoomRatio * delta);
                return true;
            }
        });
        viewFinder.setOnTouchListener((v, event) -> {
            scaleGestureDetector.onTouchEvent(event);
            return true;
        });
    }

    private void adjustZoom(float factor) {
        if (camera == null || camera.getCameraInfo().getZoomState().getValue() == null) return;
        float currentZoomRatio = camera.getCameraInfo().getZoomState().getValue().getZoomRatio();
        camera.getCameraControl().setZoomRatio(currentZoomRatio * factor);
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        }
    }

    private void showFullscreenImage(ImageView sourceImageView) {
        if (sourceImageView.getDrawable() != null) {
            imgFullscreen.setImageDrawable(sourceImageView.getDrawable());
            fullscreenContainer.setVisibility(View.VISIBLE);
        } else {
            Toast.makeText(this, "目前沒有影像可放大", Toast.LENGTH_SHORT).show();
        }
    }

    private void hideFullscreenImage() {
        fullscreenContainer.setVisibility(View.GONE);
    }

    private void toggleCamera() {
        if (!isCameraActive) {
            startCamera();
            isCameraActive = true;
            ((Button)findViewById(R.id.btnToggleCamera)).setText("Close Camera");
        } else {
            stopCamera();
            isCameraActive = false;
            ((Button)findViewById(R.id.btnToggleCamera)).setText("Open Camera📹");
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();
                ViewPort viewPort = viewFinder.getViewPort();
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                if (viewPort != null) {
                    UseCaseGroup useCaseGroup = new UseCaseGroup.Builder()
                            .addUseCase(preview)
                            .addUseCase(imageCapture)
                            .setViewPort(viewPort)
                            .build();
                    camera = cameraProvider.bindToLifecycle(this, cameraSelector, useCaseGroup);
                } else {
                    camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
                }
                txtStatus.setText("Camera Ready (Safe Mode)");
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void stopCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                cameraProvider.unbindAll();
                txtStatus.setText("Camera stopped");
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error stopping camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void takePhoto() {
        if (imageCapture == null) return;
        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull androidx.camera.core.ImageProxy image) {
                Bitmap fullBitmap = imageProxyToBitmap(image);
                image.close();
                imgOriginal.setImageBitmap(fullBitmap);
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(fullBitmap, 1024, 1024, true);
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
                inputImageBytes = stream.toByteArray();
                txtStatus.setText("Captured & Optimized (1024px)");
            }
            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Photo capture failed: " + exception.getMessage(), exception);
            }
        });
    }

    private Bitmap imageProxyToBitmap(androidx.camera.core.ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null);
        Matrix matrix = new Matrix();
        matrix.postRotate(image.getImageInfo().getRotationDegrees());
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        int width = rotatedBitmap.getWidth(), height = rotatedBitmap.getHeight();
        int newSize = Math.min(width, height);
        return Bitmap.createBitmap(rotatedBitmap, (width-newSize)/2, (height-newSize)/2, newSize, newSize);
    }

    private void processImageWithPython() {
        if (inputImageBytes == null) {
            txtStatus.setText("請先擷取圖片");
            return;
        }
        txtStatus.setText("正在分析影像...");
        new Thread(() -> {
            try {
                PyObject module = py.getModule("opencv_process");
                
                // 準備 HSV 範圍清單
                List<int[][]> hsvList = new ArrayList<>();
                for (HSVRange range : currentHsvRanges) {
                    hsvList.add(new int[][]{{range.hLow, range.sLow, range.vLow}, {range.hHigh, range.sHigh, range.vHigh}});
                }

                PyObject resultDict = module.callAttr("canny_from_image_bytes", 
                        inputImageBytes, selectedMode, 
                        (float)currentGamma, (int)currentNoiseFloor, (int)selectedTargetIndex, hsvList);

                if (resultDict == null) throw new Exception("Python 回傳空值");
                
                // Debugging: print resultDict keys
                Log.d("PythonResult", "Result keys: " + resultDict.callAttr("keys").toString());

                PyObject pyStatus = resultDict.callAttr("get", "status");
                PyObject pyImage = resultDict.callAttr("get", "image");
                if (pyStatus == null || pyImage == null || pyStatus.toString().equals("None")) {
                    String errorMsg = "格式錯誤: status=" + pyStatus + ", image=" + (pyImage != null ? "exists" : "null");
                    PyObject pyErr = resultDict.callAttr("get", "error_msg");
                    if (pyErr != null) {
                        errorMsg += " | " + pyErr.toString();
                    }
                    throw new Exception(errorMsg);
                }
                String status = pyStatus.toString();
                byte[] outPng = pyImage.toJava(byte[].class);
                runOnUiThread(() -> {
                    if (outPng != null) {
                        Bitmap outBitmap = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                        imgResult.setImageBitmap(outBitmap);
                    }
                    if ("TILTED".equals(status)) {
                        PyObject pyAngle = resultDict.callAttr("get", "angle");
                        float angle = (pyAngle != null) ? pyAngle.toJava(Float.class) : 0.0f;
                        txtStatus.setText("拍攝角度過於傾斜 (" + String.format("%.1f", angle) + "°)\n請重新拍攝");
                        txtStatus.setTextColor(android.graphics.Color.RED);
                    } else if ("ERROR".equals(status)) {
                        txtStatus.setText("Python 錯誤: " + resultDict.callAttr("get", "error_msg").toString());
                    } else if ("NO_MATCH".equals(status)) {
                        txtStatus.setText("未偵測到符合的物件");
                    } else {
                        txtStatus.setText("處理完成");
                        txtStatus.setTextColor(android.graphics.Color.GREEN);
                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> txtStatus.setText("系統異常: " + e.getMessage()));
            }
        }).start();
    }

    private void initDefaultHsvRanges() {
        currentHsvRanges.add(new HSVRange("柴犬色", 0, 32, 43, 25, 255, 255));
        currentHsvRanges.add(new HSVRange("白色", 0, 18, 150, 180, 55, 255));
    }

    private void showColorManagementDialog() {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_color_management, null);
        RecyclerView rv = view.findViewById(R.id.rvColorRanges);
        Button btnAdd = view.findViewById(R.id.btnAddColor);

        rv.setLayoutManager(new LinearLayoutManager(this));
        HSVAdapter adapter = new HSVAdapter(currentHsvRanges, this::showHsvEditDialog, this::deleteHsvRange);
        rv.setAdapter(adapter);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(view)
                .setPositiveButton("確定", null)
                .create();

        btnAdd.setOnClickListener(v -> {
            HSVRange newRange = new HSVRange("新顏色", 0, 0, 0, 180, 255, 255);
            currentHsvRanges.add(newRange);
            adapter.notifyItemInserted(currentHsvRanges.size() - 1);
            showHsvEditDialog(newRange, currentHsvRanges.size() - 1, adapter);
        });

        dialog.show();
    }

    private void deleteHsvRange(int index, HSVAdapter adapter) {
        if (index >= 0 && index < currentHsvRanges.size()) {
            currentHsvRanges.remove(index);
            adapter.notifyItemRemoved(index);
            adapter.notifyItemRangeChanged(index, currentHsvRanges.size());
        }
    }

    private void showHsvEditDialog(HSVRange range, int index, HSVAdapter adapter) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_hsv_edit, null);
        
        SeekBar sbHue = view.findViewById(R.id.sbHue);
        SeekBar sbSat = view.findViewById(R.id.sbSat);
        SeekBar sbVal = view.findViewById(R.id.sbVal);
        
        TextView tvHue = view.findViewById(R.id.tvHueValue);
        TextView tvSat = view.findViewById(R.id.tvSatValue);
        TextView tvVal = view.findViewById(R.id.tvValValue);
        
        View colorPreview = view.findViewById(R.id.viewEditColorPreview);

        // 初始化滑桿數值 (取範圍中點)
        int h = (range.hLow + range.hHigh) / 2;
        int s = (range.sLow + range.sHigh) / 2;
        int v = (range.vLow + range.vHigh) / 2;

        sbHue.setProgress(h); tvHue.setText(String.valueOf(h));
        sbSat.setProgress(s); tvSat.setText(String.valueOf(s));
        sbVal.setProgress(v); tvVal.setText(String.valueOf(v));

        SeekBar.OnSeekBarChangeListener listener = new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int ch = sbHue.getProgress();
                int cs = sbSat.getProgress();
                int cv = sbVal.getProgress();

                if (seekBar == sbHue) tvHue.setText(String.valueOf(progress));
                if (seekBar == sbSat) tvSat.setText(String.valueOf(progress));
                if (seekBar == sbVal) tvVal.setText(String.valueOf(progress));

                // 更新 Range (給予一個固定的容差值)
                range.hLow = Math.max(0, ch - 15);
                range.hHigh = Math.min(180, ch + 15);
                range.sLow = Math.max(0, cs - 50);
                range.sHigh = Math.min(255, cs + 50);
                range.vLow = Math.max(0, cv - 50);
                range.vHigh = Math.min(255, cv + 50);

                updateColorPreview(colorPreview, ch, cs, cv);
                updateSeekBarGradients(sbHue, sbSat, sbVal, ch, cs, cv);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        };

        sbHue.setOnSeekBarChangeListener(listener);
        sbSat.setOnSeekBarChangeListener(listener);
        sbVal.setOnSeekBarChangeListener(listener);

        updateColorPreview(colorPreview, h, s, v);
        updateSeekBarGradients(sbHue, sbSat, sbVal, h, s, v);

        new AlertDialog.Builder(this)
                .setTitle("編輯顏色 (HSV)")
                .setView(view)
                .setPositiveButton("完成", (d, w) -> adapter.notifyItemChanged(index))
                .show();
    }

    void updateColorPreview(View view, int h, int s, int v) {
        float[] hsv = new float[]{h * 2.0f, s / 255.0f, v / 255.0f};
        view.setBackgroundColor(Color.HSVToColor(hsv));
    }

    private void updateSeekBarGradients(SeekBar sbHue, SeekBar sbSat, SeekBar sbVal, int h, int s, int v) {
        float density = getResources().getDisplayMetrics().density;
        
        sbHue.setProgressDrawable(createFullGradientDrawable(getHueColors(), density));
        sbSat.setProgressDrawable(createFullGradientDrawable(getSatColors(h, v), density));
        sbVal.setProgressDrawable(createFullGradientDrawable(getValColors(h, s), density));
    }

    private int[] getHueColors() {
        int[] colors = new int[7];
        for (int i = 0; i < 7; i++) colors[i] = Color.HSVToColor(new float[]{i * 60f, 1f, 1f});
        return colors;
    }

    private int[] getSatColors(int h, int v) {
        return new int[]{
                Color.HSVToColor(new float[]{h * 2.0f, 0f, v / 255.0f}),
                Color.HSVToColor(new float[]{h * 2.0f, 1f, v / 255.0f})
        };
    }

    private int[] getValColors(int h, int s) {
        return new int[]{
                Color.HSVToColor(new float[]{h * 2.0f, s / 255.0f, 0f}),
                Color.HSVToColor(new float[]{h * 2.0f, s / 255.0f, 1f})
        };
    }

    private Drawable createFullGradientDrawable(int[] colors, float density) {
        GradientDrawable gd = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT, colors);
        gd.setCornerRadius(6 * density);
        
        // 關鍵：將漸層放在 background ID 下，這樣整條軌道都會顯示漸層，且不會被進度遮擋
        Drawable[] layers = new Drawable[1];
        layers[0] = gd;
        LayerDrawable ld = new LayerDrawable(layers);
        ld.setId(0, android.R.id.background);
        
        // 同時將 progress 設為透明，避免預設的進度條覆蓋漸層
        GradientDrawable transparentProgress = new GradientDrawable();
        transparentProgress.setColor(Color.TRANSPARENT);
        ld.addLayer(transparentProgress);
        ld.setId(1, android.R.id.progress);
        
        return ld;
    }

    void updateColorPreview(View view, HSVRange range) {
        // Used by adapter or other parts if needed
        int h = (range.hLow + range.hHigh) / 2;
        int s = (range.sLow + range.sHigh) / 2;
        int v = (range.vLow + range.vHigh) / 2;
        updateColorPreview(view, h, s, v);
    }

    private static class HSVAdapter extends RecyclerView.Adapter<HSVAdapter.ViewHolder> {
        private List<HSVRange> ranges;
        private OnEditListener editListener;
        private OnDeleteListener deleteListener;

        interface OnEditListener { void onEdit(HSVRange range, int index, HSVAdapter adapter); }
        interface OnDeleteListener { void onDelete(int index, HSVAdapter adapter); }

        HSVAdapter(List<HSVRange> ranges, OnEditListener editListener, OnDeleteListener deleteListener) {
            this.ranges = ranges;
            this.editListener = editListener;
            this.deleteListener = deleteListener;
        }

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_hsv_range, parent, false);
            return new ViewHolder(v);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            HSVRange range = ranges.get(position);
            holder.txtName.setText(range.name);
            holder.txtValues.setText(String.format("H:%d-%d S:%d-%d V:%d-%d", range.hLow, range.hHigh, range.sLow, range.sHigh, range.vLow, range.vHigh));
            
            // 使用外部輔助方法更新預覽顏色
            ((MainActivity)holder.itemView.getContext()).updateColorPreview(holder.colorPreview, range);

            holder.btnEdit.setOnClickListener(v -> editListener.onEdit(range, holder.getAdapterPosition(), this));
            holder.btnDelete.setOnClickListener(v -> deleteListener.onDelete(holder.getAdapterPosition(), this));
        }

        @Override
        public int getItemCount() { return ranges.size(); }

        static class ViewHolder extends RecyclerView.ViewHolder {
            TextView txtName, txtValues;
            View colorPreview;
            ImageButton btnEdit, btnDelete;
            ViewHolder(View v) {
                super(v);
                txtName = v.findViewById(R.id.txtRangeName);
                txtValues = v.findViewById(R.id.txtRangeValues);
                colorPreview = v.findViewById(R.id.viewColorPreview);
                btnEdit = v.findViewById(R.id.btnEditRange);
                btnDelete = v.findViewById(R.id.btnDeleteRange);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
        }
    }
}
