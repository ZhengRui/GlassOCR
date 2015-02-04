package com.symlab.android.glass.sharplensocr;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Locale;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.speech.tts.TextToSpeech;

import com.googlecode.leptonica.android.ReadFile;
import com.googlecode.tesseract.android.TessBaseAPI;

public class MainActivity extends Activity implements GestureDetector.OnGestureListener, GestureDetector.OnDoubleTapListener, 
Camera.OnZoomChangeListener, TextToSpeech.OnInitListener
{
	public static String TAG = "SharpLensOCR: ";
	
	public static float FULL_DISTANCE = 8000.0f;
	
    private SurfaceView mPreview;
    private SurfaceHolder mPreviewHolder;
    private Camera mCamera;
    private boolean mInPreview = false;
    private boolean mCameraConfigured = false;
    private TextView mZoomLevelView, mOcrTextView;
    private ImageView mOcrClipView;
    
    private GestureDetector mGestureDetector;
    
    
    private static final String lang = "eng";
	
	private static final String DATA_PATH = Environment
			.getExternalStorageDirectory().toString() + "/SharpLensOCR/";
	
	private TessBaseAPI baseApi;
	public static int cropWidth = 720;		// ocrCropArea width
    public static int cropHeight = 180;		// ocrCropArea Height
    public static int vBias;
    private int wbase = cropWidth;
    private int hbase = cropHeight;
    private String ocrText;
    private boolean ocrThreadRunning, ocrOnGoing, newFrame;
    private Bitmap bitmap, bitmap_to_ocr;
    DrawOnTop mDraw;
    private Mat YUVMat, RGBAMat;
    
    private TextToSpeech tts = null;
    private boolean ttsInit = false;
    
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(TAG, "OpenCV loaded successfully");
				//	YUVMat = new Mat();
				//	RGBAMat = new Mat();
				} break;
				default:
				{
					super.onManagerConnected(status);
				} break;
			}
		}
	};
    
    @Override
    public void onCreate(Bundle savedInstanceState) 
    {
    	Log.i(TAG, DATA_PATH);
    	String[] paths = new String[] { DATA_PATH, DATA_PATH + "tessdata/" };
		for (String path : paths) {
			File dir = new File(path);
			if (!dir.exists()) {
				if (!dir.mkdirs()) {
					Log.v(TAG, "ERROR: Creation of directory " + path + " on sdcard failed");
					return;
				} else {
					Log.v(TAG, "Created directory " + path + " on sdcard");
				}
			}
		}
		
		if (!(new File(DATA_PATH + "tessdata/" + lang + ".traineddata")).exists()) {
			try {
				AssetManager assetManager = getAssets();
				InputStream in = assetManager.open("tessdata/" + lang + ".traineddata");
				OutputStream out = new FileOutputStream(DATA_PATH
						+ "tessdata/" + lang + ".traineddata");
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
					out.write(buf, 0, len);
				}
				in.close();
				out.close();
				Log.v(TAG, "Copied " + lang + " traineddata");
			} catch (IOException e) {
				Log.e(TAG, "Was unable to copy " + lang + " traineddata " + e.toString());
			}
		}
		
		baseApi = new TessBaseAPI();
		baseApi.setDebug(true);
		baseApi.init(DATA_PATH, lang);
    	
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mPreview = (SurfaceView)findViewById(R.id.preview);
        mPreviewHolder = mPreview.getHolder();
        mPreviewHolder.addCallback(surfaceCallback);
        
        mZoomLevelView = (TextView)findViewById(R.id.zoomLevel);
        mOcrTextView = (TextView)findViewById(R.id.ocrText);
        mOcrClipView = (ImageView)findViewById(R.id.ocrClip);
        
        mGestureDetector = new GestureDetector(this, this);
        ocrText = "";
        
        mDraw = new DrawOnTop(this);
        addContentView(mDraw, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
        
        vBias = 0;
        tts = new TextToSpeech(this, this);
        
        
    }

	private String ocrCall(Bitmap bitmap) {
		Log.i(TAG, "OCR In");
		ocrOnGoing = true;
		baseApi.setImage(bitmap);
		
		String recognizedText = baseApi.getUTF8Text();
		
		if ( lang.equalsIgnoreCase("eng") ) {
			recognizedText = recognizedText.replaceAll("[^a-zA-Z0-9]+", " ");
		}
		
		recognizedText = recognizedText.trim();
		if (baseApi.meanConfidence() >= 50) {
			Log.i(TAG, "OCR TEXT: " + recognizedText);
			Log.i(TAG, " OCR meanConfidence: " + baseApi.meanConfidence()
					+ "\n OCR RegionBoundingBoxes: "
					+ baseApi.getRegions().getBoxRects()
					+ "\n OCR TextlineBoundingBoxes: "
					+ baseApi.getTextlines().getBoxRects()
					+ "\n OCR StripBoundingBoxes: "
					+ baseApi.getStrips().getBoxRects()
					+ "\n OCR WordBoundingBoxes: " + baseApi.getWords().getBoxRects());
		} else {
			recognizedText = "";
		}
		ocrOnGoing = false;
		newFrame = false;
		Log.i(TAG, "OCR Out");
		return recognizedText;
	}
    
    @Override
    public void onResume() 
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
        mCamera = Camera.open();
        startPreview();
        
    }

    @Override
    public void onPause() 
    {
        if ( mInPreview )
            mCamera.stopPreview();
        
        mCamera.setPreviewCallback(null);
        
        mCamera.release();
        mCamera = null;
        mInPreview = false;

        super.onPause();
    }
    
    OCRthread ocrThread = new OCRthread();
	class  OCRthread extends Thread {
		public Bitmap bitmap = null;
		public OCRthread() {
		}
		public void setBitmap(Bitmap bitmap) {
			this.bitmap = bitmap;
		}
		public void run() {
			while (ocrThreadRunning) {
				try {
					if (this.bitmap != null && newFrame) {
						ocrText = ocrCall(this.bitmap);
						if(ocrText != "") {
							Log.i(TAG, "OCR: update now");
							if(ttsInit == true)
								tts.speak(ocrText, TextToSpeech.QUEUE_FLUSH, null);
						}
					}
				} catch (Exception e) {
					Log.i(TAG, "OCRED Not called as " + e.toString());
				}
			}
		}
	}
    
    
    
    private void initPreview(int width, int height) 
    {
        if ( mCamera != null && mPreviewHolder.getSurface() != null) {
            try 
            {
                mCamera.setPreviewDisplay(mPreviewHolder);
                mCamera.setPreviewCallback(new PreviewCallback() {
					
					@Override
					public void onPreviewFrame(byte[] data, Camera camera) {
						if (!ocrOnGoing) {
						Log.i(TAG, " onPreviewFrame() called.");
						Log.i(TAG, " data length: " + data.length + " camera size: " + 
								camera.getParameters().getPreviewSize().width + ", " + camera.getParameters().getPreviewSize().height);

						
						// for debuging
					    String dbgBmpDirPath = DATA_PATH + "DebugBmp";
					    File dir = new File(dbgBmpDirPath);
					    if (!dir.exists())
					    	dir.mkdirs();
						
						
					    // Manually decoding
						
					    Size size = camera.getParameters().getPreviewSize();
					    int dataWidth = size.width;
					    int dataHeight = size.height;
					    int[] pixels = new int[cropWidth * cropHeight];
					    
						byte[] yuv = data;
					    int inputOffset = ((dataHeight - cropHeight) / 2 + vBias) * dataWidth + (dataWidth - cropWidth) / 2;
					    int iOffset = (dataHeight - cropHeight) / 2;
					    int jOffset = (dataWidth - cropWidth) / 2;
					    int framesize = dataWidth * dataHeight;

					    
					    // Only gray
					    
					    
					    for (int i = 0; i < cropHeight; i++) {
					      int outputOffset = i * cropWidth;
					      for (int j = 0; j < cropWidth; j++) {
					        int grey = yuv[inputOffset + j] & 0xff;
					        pixels[outputOffset + j] = 0xFF000000 | (grey * 0x00010101);
					      }
					      inputOffset += dataWidth;
					    }
					    
					    
					    
					    // With Color
					    /*
					    for (int i = 0; i < cropHeight; i++) {
					    	int outputOffset = i * cropWidth;
					    	int i_ori = i + iOffset + vBias;
					    	for (int j = 0; j < cropWidth; j++) {
					    		int j_ori = j + jOffset;
					    		int y = yuv[i_ori * dataWidth + j_ori] & 0xff;
					    		int u = yuv[framesize + (i_ori >> 1) * dataWidth + (j_ori & ~1) + 0] & 0xff;
					    		int v = yuv[framesize + (i_ori >> 1) * dataWidth + (j_ori & ~1) + 1] & 0xff;
					    		y = y < 16 ? 16 : y;

			                    int r = Math.round(1.164f * (y - 16) + 1.596f * (v - 128));
			                    int g = Math.round(1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));
			                    int b = Math.round(1.164f * (y - 16) + 2.018f * (u - 128));

			                    r = r < 0 ? 0 : (r > 255 ? 255 : r);
			                    g = g < 0 ? 0 : (g > 255 ? 255 : g);
			                    b = b < 0 ? 0 : (b > 255 ? 255 : b);
			                    pixels[outputOffset + j] = 0xff000000 + (b << 16) + (g << 8) + r;
					    	}
					    }
					    */
					    
					    
					    bitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);
					    bitmap.setPixels(pixels, 0, cropWidth, 0, 0, cropWidth, cropHeight);
					    
					
						/*
						// Using BitFactory:
						// Convert to JPG, then bitmap
						Size previewSize = camera.getParameters().getPreviewSize(); 
						YuvImage yuvimage=new YuvImage(data, ImageFormat.NV21, previewSize.width, previewSize.height, null);
						ByteArrayOutputStream baos = new ByteArrayOutputStream();
						yuvimage.compressToJpeg(new Rect(0, 0, previewSize.width, previewSize.height), 100, baos);
						byte[] jdata = baos.toByteArray();
						BitmapFactory.Options bitmapFatoryOptions = new BitmapFactory.Options();
			            bitmapFatoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
						Bitmap bitmap = BitmapFactory.decodeByteArray(jdata, 0, jdata.length, bitmapFatoryOptions);
						Log.i(TAG, " jdata length: " + jdata.length);
						
						try {
							FileOutputStream jpgOut;
							File dbgJpg = new File(dir, "dbg.jpg");
							jpgOut = new FileOutputStream(dbgJpg);
							try {
								jpgOut.write(jdata);
								jpgOut.flush();
								jpgOut.close();
							} catch (IOException e1) {
								e1.printStackTrace();
							}	
						} catch (FileNotFoundException e) {
							e.printStackTrace();
						}
						*/
					    
					    /*
					    try {
							File dbgBmp = new File(dir, "dbg.bmp");
					    	FileOutputStream bmpOut;
							bmpOut = new FileOutputStream(dbgBmp);
					    	bitmap.compress(Bitmap.CompressFormat.PNG, 100, bmpOut);
					    	try {
								bmpOut.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
					    	try {
								bmpOut.close();
							} catch (IOException e) {
								e.printStackTrace();
							}
						} catch (FileNotFoundException e) {
							e.printStackTrace();
						}
					    
						Log.i(TAG, " dbg.bmp and dbg.jpg saved to " + dbgBmpDirPath);
					    */
					    
					    
					    
					    bitmap_to_ocr = Bitmap.createScaledBitmap(bitmap, 300, 45, true);
					    
					    /*
					    
					    // Get histograme of each frame thumbnail
					    int[] hist = new int[255];
					    for (int i = 0; i < bitmap_to_ocr.getWidth(); i++) {
					    	for (int j = 0; j < bitmap_to_ocr.getHeight(); j++) {
					    		int col = bitmap_to_ocr.getPixel(i, j);  
					            int alpha = col&0xFF000000;  
					            int red = (col&0x00FF0000)>>16;  
					            int green = (col&0x0000FF00)>>8;  
					            int blue = (col&0x000000FF);  
					            int gray = (int)((float)red*0.3+(float)green*0.59+(float)blue*0.11);
					            hist[gray]++;
					    	}
					    }
					    					    
					    Log.i(TAG, "Frame Histogram: " + Arrays.toString(hist));
					    
					    */
					    
					    // data[] array to opecvMat YUV format, then to opencvMat BGR format				    
					    /*
					    Log.i(TAG, "CV Starts");
					    
					    YUVMat = new Mat(dataHeight + dataHeight / 2, dataWidth, CvType.CV_8UC1);
					    YUVMat.put(0, 0, data);
					    RGBAMat = new Mat(dataHeight, dataWidth, CvType.CV_8UC4);
					    Imgproc.cvtColor(YUVMat, RGBAMat, Imgproc.COLOR_YUV420sp2RGBA);
					    
					    bitmap_to_ocr = Bitmap.createBitmap(dataWidth, dataHeight, Bitmap.Config.ARGB_8888); 
					    Utils.matToBitmap(RGBAMat, bitmap_to_ocr, true);
					    
					    Log.i(TAG, "CV Ends");
					    */
					    
					   // bitmap_to_ocr = bitmap;
					    
					    mOcrClipView.setImageBitmap(bitmap_to_ocr);
					    
					    
						
					    /*
					    ocrThread.setBitmap(bitmap_to_ocr);
					    newFrame = true;
						mOcrTextView.setText("OCR: " + ocrText);
						*/
					    
					    
					    
					    
					    
						}
					} 
				});
            }
            catch (Throwable t) 
            {
                Log.e(TAG, "Exception in initPreview()", t);
                Toast.makeText(MainActivity.this, t.getMessage(), Toast.LENGTH_LONG).show();
            }

            if ( !mCameraConfigured ) 
            {
                Camera.Parameters parameters = mCamera.getParameters();
                parameters.setPreviewFpsRange(30000, 30000);
                parameters.setPreviewSize(1920, 1080); // hard coded the largest size for now
                mCamera.setParameters(parameters);
                mCamera.setZoomChangeListener(this);
                
                mCameraConfigured = true;
            }
        }
    }

    private void startPreview() 
    {
        if ( mCameraConfigured && mCamera != null ) 
        {
            mCamera.startPreview();
            mInPreview = true;
        }
    }
    
  
    public void onPreviewFrame(byte[] data, Camera cam) {
    	
    }
    
    
    SurfaceHolder.Callback surfaceCallback = new SurfaceHolder.Callback() {
        public void surfaceCreated( SurfaceHolder holder ) 
        {
        	ocrThreadRunning = true;
        	ocrOnGoing = false;
        	newFrame = false;
        	ocrThread.start();
        }

        public void surfaceChanged( SurfaceHolder holder, int format, int width, int height ) 
        {
        	ocrThreadRunning = false;
            initPreview(width, height);
            startPreview();
        }

        public void surfaceDestroyed( SurfaceHolder holder ) 
        {
            ocrThreadRunning = false;
        }
    };
    
    @Override
    public boolean onGenericMotionEvent(MotionEvent event) 
    {
        mGestureDetector.onTouchEvent(event);
        return true;
    }
    
    

	@Override
	public boolean onDown(MotionEvent e) 
	{
	//	Log.i(TAG, " onDown() called");
		
		return false;
	}
	
	@Override
	public boolean onFling( MotionEvent e1, MotionEvent e2, float velocityX, float velocityY ) 
	{
		Log.d(TAG, "velocityX: " + velocityX + ", velocityY: " + velocityY);
		
		if (velocityX > 1f)
		{
			vBias = (vBias + 50) > (1000 - cropHeight) / 2 ? (cropHeight - 1000) / 2 : (vBias + 50);
		}
		else if (velocityX < -1f)
		{
			vBias = (vBias - 50) < (cropHeight - 1000) / 2 ? (1000 - cropHeight) / 2 : (vBias - 50);
		}
		
		mDraw.invalidate();
		return false;
	}
	
	
	@Override
	public boolean onScroll( MotionEvent e1, MotionEvent e2, float distanceX, float distanceY ) 
	{
		Log.d(TAG, "distanceX: " + distanceX + ", distanceY: " + distanceY);
		return false;
	}
	
	
	@Override
	public void onLongPress(MotionEvent e) 
	{
	//	Log.i(TAG, " onLongPress() called");
		ocrThreadRunning = false;
		tts.stop();
		tts.shutdown();
		finish();
	}

	@Override
	public void onShowPress(MotionEvent e) 
	{
		
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e) 
	{
		return false;
	}
	
	@Override
	public void onZoomChange(int zoomValue, boolean stopped, Camera camera) {
		mZoomLevelView.setText("ZOOM: " + zoomValue);
		
	}

	@Override
	public boolean onDoubleTap(MotionEvent e) {
		Camera.Parameters parameters = mCamera.getParameters();
	    parameters.setPreviewFpsRange(30000, 30000);
		int zoom = parameters.getZoom();
		zoom -= 5;
		if ( zoom < 0 )
			zoom = 0;
		mCamera.startSmoothZoom(zoom);
		cropWidth = wbase + (1800 - wbase) * (zoom - 1) / 59;
		cropHeight = hbase + (360 - hbase) * (zoom - 1) / 59;
		mDraw.invalidate();
		return false;
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e) {
		return false;
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e) {
		Camera.Parameters parameters = mCamera.getParameters();
	    parameters.setPreviewFpsRange(30000, 30000);
		int zoom = parameters.getZoom();
		zoom += 5;
		if ( zoom > parameters.getMaxZoom() )
			zoom = parameters.getMaxZoom();
		mCamera.startSmoothZoom(zoom);
		cropWidth = wbase + (1800 - wbase) * (zoom - 1) / 59;
		cropHeight = hbase + (360 - hbase) * (zoom - 1) / 59;
		mDraw.invalidate();	
		return false;
	}
	
	@Override
	public void onInit(int arg0) {
		
		if(arg0 == TextToSpeech.SUCCESS){
			ttsInit = true;
			tts.setLanguage(Locale.US);
		}
	}
	
}

class DrawOnTop extends View {
	 
    public DrawOnTop(Context context) {
            super(context);
    }

    @SuppressLint("DrawAllocation")
	@Override
    protected void onDraw(Canvas canvas) {

            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.BLACK);
            canvas.drawRect(320 - MainActivity.cropWidth / 6, 180 - MainActivity.cropHeight / 6 + MainActivity.vBias / 3, 320 + MainActivity.cropWidth / 6,  180 + MainActivity.cropHeight / 6 + MainActivity.vBias / 3, paint);
            
            super.onDraw(canvas);

    }
}


