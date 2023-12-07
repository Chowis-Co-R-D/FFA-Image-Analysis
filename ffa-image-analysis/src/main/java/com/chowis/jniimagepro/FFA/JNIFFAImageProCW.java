package com.chowis.jniimagepro.FFA;

public class JNIFFAImageProCW
{

	static
	{
		System.loadLibrary("JNIFFAImageProCWCore");
	}

	public native String getVersionJni();
	public native String getMakeDateJni();
    
    // FFA Local algorithms.
    // 2023-04-04, updated by Shu Li.
    public native String FFALocalOiliness100Jni(
    		String pplFrontOriginalImgPath,
    		String oilinessRoiImgPath,
    		String resultImgOutputPath,
    		String greenMaskImgOutputPath,
    		String whiteMaskImgOutputPath);
    
    public native String FFALocalRadianceDullness100Jni(
    		String pplFrontOriginalImgPath, 
    		String radianceDullnessFrontRoiImgPath, 
    		String resultImgOutputPath, 
    		String grayMaskImgOutputPath, 
			String whiteMaskImgOutputPath);
    
    public native String FFALocalPores100Jni(
            String poresRoiImgPath,
            String pplFrontOriginalImgPath,
            String resultImgOutputPath,
            String maskImgOutputPath);
    
    public native String FFALocalPigmentationSpots100Jni(
            String frontOriginalInputPath,
            String leftOriginalInputPath,
            String rightOriginalInputPath,
            String frontResultOutputPath,
            String leftResultOutputPath,
            String rightResultOutputPath,
            String frontForeheadMaskInputPath,
            String frontNoseMaskInputPath,
            String frontCheekMaskInputPath,
            String frontChinMaskInputPath,
            String frontFullFaceROIMaskInputPath,
            String leftForeheadMaskInputPath,
            String leftNoseMaskInputPath,
            String leftCheekMaskInputPath,
            String leftChinMaskInputPath,
            String leftFullFaceROIMaskInputPath,
            String rightForeheadMaskInputPath,
            String rightNoseMaskInputPath,
            String rightCheekMaskInputPath,
            String rightChinMaskInputPath,
            String rightFullFaceROIMaskInputPath,
            String frontFullSpotsMaskInputPath,
            String leftFullSpotsMaskInputPath,
            String rightFullSpotsMaskInputPath,
            boolean pigmentationSpotsSideImageEnabled);
	
    public native double FFALocalElasticity100Jni(double wrinkleScore, double dullnessScore);
    
    public native double FFAGetAnalyzedImgJni(String inputOriginalImgPath, String inputMaskImgPath, String outputAnalyzedImgPath, double alpha, int maskB, int maskG, int maskR, int contourB, int contourG, int contourR);
}
