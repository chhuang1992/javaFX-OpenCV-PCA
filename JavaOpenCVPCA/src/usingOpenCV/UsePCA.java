package usingOpenCV;

import java.util.ArrayList;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

//http://docs.opencv.org/3.1.0/d1/dee/tutorial_introduction_to_pca.html
public class UsePCA {
	
	public static void drawAxis( Mat img, Point p, Point q, Scalar colour, double scale){
		double angle;
	    double hypotenuse;
	    angle = Math.atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
	    hypotenuse = Math.sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	    
	    Point q1 = new Point();
	    Point p1 = new Point();
	    Point p2 = new Point();
	    q1.x = (int) (p.x - scale * hypotenuse * Math.cos(angle));
	    q1.y = (int) (p.y - scale * hypotenuse * Math.sin(angle));
	    //line(img, p, q, colour, 1, CV_AA)  in C++;
	    Core.line(img, p, q1, colour, 1, Core.LINE_AA,0);
	    
	    
	    p1.x = (int) (q1.x + 9 * Math.cos(angle + Math.PI / 4));
	    p1.y = (int) (q1.y + 9 * Math.sin(angle + Math.PI / 4));
	    Core.line(img, p1, q1, colour, 1, Core.LINE_AA,0);

	    p2.x = (int) (q1.x + 9 * Math.cos(angle - Math.PI / 4));
	    p2.y = (int) (q1.y + 9 * Math.sin(angle - Math.PI / 4));
	    Core.line(img, p2, q1, colour, 1, Core.LINE_AA,0);
	}
	
	public static double getOrientation(MatOfPoint pts_, Mat img){
		Point[] pts = pts_.toArray();
		int sz = pts.length;
		Mat data_pts =new Mat(sz, 2, CvType.CV_64FC1);
		
		for (int i = 0; i < data_pts.rows(); ++i){
			data_pts.put(i, 0, pts[i].x);
			data_pts.put(i, 1, pts[i].y);
		}
				
		Mat mean = new Mat();
		Mat eigenvalues = new Mat();
		Mat eigenvectors = new Mat();
		Mat vectors = new Mat();
		Mat covar = new Mat();
		Core.PCACompute(data_pts, mean, vectors);
		System.out.println("Using PCACompute to compute InputData's eigenvector, we get:");
		System.out.println(vectors.dump());
		
		Core.calcCovarMatrix(data_pts, covar, mean, Core.COVAR_NORMAL | Core.COVAR_SCALE | Core.COVAR_ROWS | Core.COVAR_USE_AVG);
		
		Core.eigen(covar, true, eigenvalues, eigenvectors);
		System.out.println("The InputData's Covariance Matrix's eigenvector and eigenvalue:");
		System.out.println(eigenvectors.dump());
		System.out.println(eigenvalues.dump());
		
		Point cntr = new Point(mean.get(0, 0)[0],mean.get(0, 1)[0]);
		
		Point[] eigen_vecs =new Point[2];
		Double[] eigen_val =new Double[2];
                
		for (int i = 0; i < 2; ++i)
	    {
			eigen_vecs[i]=new Point(eigenvectors.get(i, 0)[0], eigenvectors.get(i, 1)[0]);
	        eigen_val[i]=eigenvalues.get(i,0)[0];
	    }
		
		Core.circle(img, cntr, 3, new Scalar(255, 0, 255), 2);
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

	    return angle;
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat src = Highgui.imread("data/pca_test1.jpg");
		
		// Convert image to grayscale
		Mat gray = new Mat();
	    Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
		
	    // Convert image to binary
	    Mat bw = new Mat();
	    Imgproc.threshold(gray, bw, 50, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
	    
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
	        Imgproc.drawContours(src, contours, i,new Scalar(0, 0, 255), 2);
	        // Find the orientation of each shape
	        getOrientation(contours.get(i), src);
	    }
	    	    
	    Highgui.imwrite("data/PCAresult.png",src);
	}
}
