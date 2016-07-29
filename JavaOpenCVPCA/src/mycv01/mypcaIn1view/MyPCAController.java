package mycv01.mypcaIn1view;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
//import java.util.List;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
//import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;  //opencv.imgcodecs.Imgcodecs in OpenCV3
import org.opencv.imgproc.Imgproc;

//import org.opencv.imgcodecs.Imgcodecs;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

/**
 * The controller associated to the only view of our application. The
 * application logic is implemented here. It handles the button for opening an
 * image and perform all the operation related to the PCA Compute.
 * 
 * @author <a href="mailto:chhuang1992@gmail.com">Huangchunhui (from China)</a>
 * @version 1.0 (2016-7-28)
 * @reference 
 * <a href="https://github.com/opencv-java/">OpenCV with Java(FX) by Luigi De Russis(Italy)</a>
 * <a href="http://docs.opencv.org/3.1.0/d1/dee/tutorial_introduction_to_pca.html">
 *     opencv tutorials about PCA
 * </a>
 * <a href="http://docs.opencv.org/java/2.4.4/">Opencv java API</a>
 * 
 * @implements   win7 X64; JavaSE1.8; JavaFX; Eclipse Mars; opencv-java2.4
 */
public class MyPCAController
{
	// images to show in the view
	@FXML
	private ImageView currentImage;
	@FXML
	private ImageView histogram;
	@FXML
	private ImageView binImage;
	
	// a FXML button for performing the transformation	
	@FXML
	private Button loadButton;
	// a FXML button for performing the transformation	
	@FXML
	private Button transformButton;
	// a FXML button for performing the antitransformation
	@FXML
	private Button pcaButton;
	
	// the FXML grayscale checkbox
	@FXML
	private CheckBox grayscale;
	
	@FXML
	private Slider valueStart;
	@FXML
	private Slider valueStop;
	
	// the main stage
	private Stage stage;
	// the JavaFX file chooser
	private FileChooser fileChooser;
	// support variables
	private Mat image;
	// binary image;
	private Mat bw;
	
	/**
	 * Init the needed variables
	 */
	protected void init()
	{
		this.fileChooser = new FileChooser();
		this.image = new Mat();
		this.bw = new Mat();
	}
	
	/**
	 * Load an image from disk.
	 * Come from OpenCV with Java(FX) by Luigi De Russis(Italy).
	 */
	@FXML
	protected void loadImage()
	{
		// show the open dialog window
		File file = this.fileChooser.showOpenDialog(this.stage);
		if (file != null)
		{
			// read the image in gray scale
			String filepath = file.getAbsolutePath();
			this.image = Highgui.imread(filepath);
			// show the image
			this.currentImage.setImage(this.mat2Image(this.image));
			//this.originalImage.setImage(this.mat2Image(this.image));
			
			// set a fixed width
			this.currentImage.setFitWidth(600);
			// preserve image ratio
			this.currentImage.setPreserveRatio(true);
			//
			this.showHistogram(this.image, grayscale.isSelected());
			// update the UI
			this.transformButton.setDisable(false);
		}
	}
	
	/**
	 * The action triggered by pushing the button for apply the binary transform to the loaded image.
	 * @reference opencv tutorials about PCA
	 */
	@FXML
	protected void transformImage()
	{
		// Convert image to grayscale
		Mat gray = new Mat();
	    Imgproc.cvtColor(this.image, gray, Imgproc.COLOR_BGR2GRAY);
	    
	    // Convert image to binary
	    Imgproc.threshold(gray, this.bw, this.valueStart.getValue(), this.valueStop.getValue(), Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
		
		// show the result of the transformation as an image.
		this.binImage.setImage(this.mat2Image(this.bw));
		
		// set a fixed width
		this.binImage.setFitWidth(150);
		// preserve image ratio
		this.binImage.setPreserveRatio(true);
		
		// enable the button for performing the PCA
		this.pcaButton.setDisable(false);
		// disable the button for applying the binary transform.
		this.transformButton.setDisable(true);
	}
	
	/**
	 * The action triggered by pushing the button for apply the PCA to the loaded image
	 * @reference opencv tutorials about PCA
	 */
	@FXML
	protected void pcaImage()
	{
		Mat hierarchy = new Mat();
	    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint> ();
	    Imgproc.findContours(bw, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		
	    for (int i = 0; i < contours.size(); ++i)
	    {
	        // Calculate the area of each contour
	        double area = Imgproc.contourArea(contours.get(i));
	        // Ignore contours that are too small or too large
	        if (area < 1e2 || 1e5 < area) continue;

	        // Draw each contour only for visualisation purposes
	        Imgproc.drawContours(this.image, contours, i,new Scalar(0, 0, 255), 2);
	        // Find the orientation of each shape
	        getOrientation(contours.get(i), image);
	    }
		
		this.currentImage.setImage(this.mat2Image(this.image));
		// set a fixed width
		this.currentImage.setFitWidth(600);
		// preserve image ratio
		this.currentImage.setPreserveRatio(true);
		
		// disable the button for performing the PCA
		this.pcaButton.setDisable(true);
	}
	
	/**
	 * Set the current stage (needed for the FileChooser modal window)
	 * 
	 * @param stage
	 *            the stage
	 * Come from OpenCV with Java(FX) by Luigi De Russis(Italy).
	 */
	public void setStage(Stage stage)
	{
		this.stage = stage;
	}
		
	/**
	 * @function static drawAxis
	 * draw lines to {@argument Mat img}   Call by pcaImage.
	 * void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)  in C++;
	 */
	public static void drawAxis( Mat img, Point p, Point q, Scalar colour, double scale){
		double angle;
	    double hypotenuse;
	    angle = Math.atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
	    hypotenuse = Math.sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	    
	    //When implement Point, java use its reference value, it will change the arguement out of function.
	    //So, I define the three Point p1,p2,q1 to avoid use p, q.
	    //In C++, we need not to define the three Point param.
	    Point q1 = new Point();
	    Point p1 = new Point();
	    Point p2 = new Point();
	    q1.x = (int) (p.x - scale * hypotenuse * Math.cos(angle));
	    q1.y = (int) (p.y - scale * hypotenuse * Math.sin(angle));
	    //line(img, p, q, colour, 1, CV_AA);  in C++;
	    //Imgproc.line(img, p, q, colour, 1, Core.LINE_AA,0);  using OpenCV3
	    Core.line(img, p, q1, colour, 1, Core.LINE_AA,0);
	    	    
	    p1.x = (int) (q1.x + 9 * Math.cos(angle + Math.PI / 4));
	    p1.y = (int) (q1.y + 9 * Math.sin(angle + Math.PI / 4));
	    Core.line(img, p1, q1, colour, 1, Core.LINE_AA,0);

	    p2.x = (int) (q1.x + 9 * Math.cos(angle - Math.PI / 4));
	    p2.y = (int) (q1.y + 9 * Math.sin(angle - Math.PI / 4));
	    Core.line(img, p2, q1, colour, 1, Core.LINE_AA,0);
	}
		
	/**
	 * @function static getOrientation
	 * Compute mean, covariance matrix, pca of {@param Mat img}
	 * 
	 * @argument Mat
	 *            draw the {@link Contours} to the current Mat image     Call by pcaImage.
	 * double getOrientation(const vector<Point> &pts, Mat &img);   in C++
	 */
	public static double getOrientation(MatOfPoint pts_, Mat img){
		Point[] pts = pts_.toArray();
		int sz = pts.length;
		Mat data_pts =new Mat(sz, 2, CvType.CV_64FC1);
		
		for (int i = 0; i < data_pts.rows(); ++i){
			data_pts.put(i, 0, pts[i].x);
			data_pts.put(i, 1, pts[i].y);
		}
		
		//in C++ we can use PCA class to perform PCA analysis;
		//PCA class has member variable eigenvalues, eigenvectors,
		//so we don't need to perform PCA analysis like this way.
		//PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW); in C++
		//perform PCA analysis in java
		Mat mean = new Mat();    //PCA analysis begin;
		Mat eigenvalues = new Mat();
		Mat eigenvectors = new Mat();
		Mat vectors = new Mat();
		Mat covar = new Mat();
		Core.PCACompute(data_pts, mean, vectors);
		
		/**
		 * the local java operation related to the PCA Compute can't get eigenvalues;
		 * so, we read the PCA class sourse file in C++;
		 * we can know that eigenvalues and eigenvectors is data's Covariance Matrix's;
		 * calcCovarMatrix's CovarFlag is initial to COVAR_SCALA
		 * WHEN PCA::Flag = DATA_AS_ROW    THEN  CovarFlag |= COLAR_ROWS
		 * WHEN cols<rows    THEN  CovarFlag |= COLAR_NORMAL
		 * WHEN data is not empty   THEN  CovarFlag |= COLAR_USE_AVG
		 * SO finally CovarFlag = Core.COVAR_NORMAL | Core.COVAR_USE_AVG | Core.COVAR_SCALE | Core.COVAR_ROWS
		 */
		Core.calcCovarMatrix(data_pts, covar, mean, Core.COVAR_NORMAL | Core.COVAR_USE_AVG | Core.COVAR_SCALE | Core.COVAR_ROWS);
		
		Core.eigen(covar, true, eigenvalues, eigenvectors);
		System.out.println("PCA's eigenvector and eigenvalue:");
		System.out.println(eigenvectors.dump());
		System.out.println(eigenvalues.dump());//PCA analysis end;
		
		
	//visualization
		
		//visualization start
		/**
		 * Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
         *                   static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
         * in C++;
         */
		//Store the center of the object
		Point cntr = new Point(mean.get(0, 0)[0],mean.get(0, 1)[0]);
		
		/**
		 * vector<Point2d> eigen_vecs(2);
    	 * vector<double> eigen_val(2);
         * for (int i = 0; i < 2; ++i)
    	 * {
         * 		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
         *                        		pca_analysis.eigenvectors.at<double>(i, 1));

         * 		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    	 * }
         * in C++;
         */
		//Store the eigenvalues and eigenvectors
		Point[] eigen_vecs =new Point[2];    //We can alse use ArrayList<Point>;
		Double[] eigen_val =new Double[2];   //We can alse use ArrayList<Double>;                
		for (int i = 0; i < 2; ++i)
	    {
			eigen_vecs[i]=new Point(eigenvectors.get(i, 0)[0], eigenvectors.get(i, 1)[0]);
	        eigen_val[i]=eigenvalues.get(i,0)[0];
	    }
		
		// Draw the principal components
		// circle(img, cntr, 3, Scalar(255, 0, 255), 2); in C++;
		Core.circle(img, cntr, 3, new Scalar(255, 0, 255), 2);  //Imageproc.circle in OpenCV3
		/**
		 * Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), 
		 *                                static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    	 * Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), 
    	 *                                static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    	 * in C++; C++ can implement operator overload; so it can directly use "+";
		 */
		Point p1_ = new Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]);
		Point p1 = new Point(cntr.x+0.02*p1_.x,cntr.y+0.02*p1_.y);
		Point p2_ = new Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]);
		Point p2 = new Point(cntr.x-0.02*p2_.x,cntr.y-0.02*p2_.y);		
		//drawAxis is a mothod operating variable by its reference,so copy Point to reserve data.
		Point cntr1 = cntr.clone();
		Point cntr2 = cntr.clone();		
		drawAxis(img, cntr1, p1, new Scalar(0, 255, 0), 1);
	    drawAxis(img, cntr2, p2, new Scalar(255, 255, 0), 5);

	    double angle = Math.atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
	//visualization end

	    return angle;
	}
	/**
	 * Compute and show the histogram for the given {@link Mat} image
	 * 
	 * @param frame
	 *            the {@link Mat} image for which compute the histogram
	 * @param gray
	 *            is a grayscale image?
	 * @reference Come from OpenCV with Java(FX) by Luigi De Russis(Italy).
	 */
	private void showHistogram(Mat frame, boolean gray)
	{
		// split the frames in multiple images
		List<Mat> images = new ArrayList<Mat>();
		Core.split(frame, images);
		
		// set the number of bins at 256
		MatOfInt histSize = new MatOfInt(256);
		// only one channel
		MatOfInt channels = new MatOfInt(0);
		// set the ranges
		MatOfFloat histRange = new MatOfFloat(0, 256);
		
		// compute the histograms for the B, G and R components
		Mat hist_b = new Mat();
		Mat hist_g = new Mat();
		Mat hist_r = new Mat();
		
		// B component or gray image
		Imgproc.calcHist(images.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange, false);
		
		// G and R components (if the image is not in gray scale)
		if (!gray)
		{
			Imgproc.calcHist(images.subList(1, 2), channels, new Mat(), hist_g, histSize, histRange, false);
			Imgproc.calcHist(images.subList(2, 3), channels, new Mat(), hist_r, histSize, histRange, false);
		}
		
		// draw the histogram
		int hist_w = 150; // width of the histogram image
		int hist_h = 150; // height of the histogram image
		int bin_w = (int) Math.round(hist_w / histSize.get(0, 0)[0]);
		
		Mat histImage = new Mat(hist_h, hist_w, CvType.CV_8UC3, new Scalar(0, 0, 0));
		// normalize the result to [0, histImage.rows()]
		Core.normalize(hist_b, hist_b, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
		
		// for G and R components
		if (!gray)
		{
			Core.normalize(hist_g, hist_g, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
			Core.normalize(hist_r, hist_r, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
		}
		
		// effectively draw the histogram(s)
		for (int i = 1; i < histSize.get(0, 0)[0]; i++)
		{
			// B component or gray image
			Core.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_b.get(i - 1, 0)[0])),
					new Point(bin_w * (i), hist_h - Math.round(hist_b.get(i, 0)[0])), new Scalar(255, 0, 0), 2, 8, 0);
			// G and R components (if the image is not in gray scale)
			if (!gray)
			{
				Core.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_g.get(i - 1, 0)[0])),
						new Point(bin_w * (i), hist_h - Math.round(hist_g.get(i, 0)[0])), new Scalar(0, 255, 0), 2, 8,
						0);
				Core.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hist_r.get(i - 1, 0)[0])),
						new Point(bin_w * (i), hist_h - Math.round(hist_r.get(i, 0)[0])), new Scalar(0, 0, 255), 2, 8,
						0);
			}
		}
		
		// display the histogram...
		Image histImg = mat2Image(histImage);
		this.histogram.setImage(histImg);

	}
	
	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 * @reference Come from OpenCV with Java(FX) by Luigi De Russis(Italy).
	 */
	private Image mat2Image(Mat frame)
	{
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Highgui.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}
}
