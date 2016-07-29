package mycv01.mypcaIn1view;

import org.opencv.core.Core;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;

/**
 * The main class for a JavaFX application. It creates and handle the main
 * window with its resources (style, graphics, etc.).
 * 
 * This application opens an image stored on disk and perform the PCA Compute.
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
public class MyPCAIn1View extends Application
{
	// the main stage
	private Stage primaryStage;
	
	@Override
	public void start(Stage primaryStage)
	{
		try
		{
			// load the FXML resource
			FXMLLoader loader = new FXMLLoader(getClass().getResource("MyPCA.fxml"));
			BorderPane root = (BorderPane) loader.load();
			// set a whitesmoke background
			root.setStyle("-fx-background-color: whitesmoke;");
			Scene scene = new Scene(root, 800, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			// create the stage with the given title and the previously created scene
			this.primaryStage = primaryStage;
			this.primaryStage.setTitle("PCACompute");
			this.primaryStage.setScene(scene);
			this.primaryStage.show();
			
			// init the controller
			MyPCAController controller = loader.getController();
			controller.setStage(this.primaryStage);
			controller.init();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args)
	{
		// load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		launch(args);
	}
}
