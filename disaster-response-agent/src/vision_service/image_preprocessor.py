import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging
from typing import Tuple, Optional, Dict, Any
import base64

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def preprocess_image(self, image_data: bytes, enhance: bool = True) -> bytes:
        """
        Main preprocessing pipeline for disaster images
        
        Args:
            image_data: Raw image bytes
            enhance: Whether to apply enhancement filters
            
        Returns:
            Preprocessed image bytes
        """
        try:
            # Load image
            image = self._load_image_from_bytes(image_data)
            
            # Validate image
            if not self._validate_image(image):
                raise ValueError("Invalid or corrupted image")
            
            # Resize image
            image = self._resize_image(image)
            
            # Enhance image quality if requested
            if enhance:
                image = self._enhance_image(image)
            
            # Normalize image
            image = self._normalize_image(image)
            
            # Convert back to bytes
            return self._image_to_bytes(image)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _load_image_from_bytes(self, image_data: bytes) -> Image.Image:
        """Load PIL Image from bytes"""
        try:
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError("Invalid image format")
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate image properties"""
        try:
            # Check if image is too small
            min_size = 64
            if image.width < min_size or image.height < min_size:
                logger.warning(f"Image too small: {image.width}x{image.height}")
                return False
            
            # Check if image is too large
            max_size = 4096
            if image.width > max_size or image.height > max_size:
                logger.warning(f"Image too large: {image.width}x{image.height}")
                # Don't fail, just log warning - we'll resize it
            
            # Check if image has valid data
            if not hasattr(image, 'getdata'):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        try:
            # Calculate new size maintaining aspect ratio
            original_width, original_height = image.size
            target_width, target_height = self.target_size
            
            # Calculate scaling factor
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            
            # Resize image with high quality resampling
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image centered
            final_image = Image.new('RGB', self.target_size, (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            return final_image
            
        except Exception as e:
            logger.error(f"Image resizing failed: {e}")
            # Return original image if resizing fails
            return image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhancement filters for better AI analysis"""
        try:
            enhanced_image = image.copy()
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = contrast_enhancer.enhance(1.2)  # 20% more contrast
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = sharpness_enhancer.enhance(1.1)  # 10% more sharpness
            
            # Enhance color saturation slightly
            color_enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = color_enhancer.enhance(1.1)  # 10% more saturation
            
            # Apply slight noise reduction
            enhanced_image = enhanced_image.filter(ImageFilter.SMOOTH_MORE)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize image for consistent AI processing"""
        try:
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array / 255.0
            
            # Apply histogram equalization for better contrast
            img_array = self._histogram_equalization(img_array)
            
            # Convert back to [0, 255] range
            img_array = (img_array * 255).astype(np.uint8)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            return image
    
    def _histogram_equalization(self, img_array: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast"""
        try:
            # Convert to grayscale for histogram calculation
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Calculate histogram
            hist, bins = np.histogram(gray.flatten(), 256, [0, 1])
            
            # Calculate cumulative distribution function
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            
            # Apply equalization if needed (only if image is too dark/bright)
            mean_brightness = np.mean(gray)
            if mean_brightness < 0.3 or mean_brightness > 0.7:
                # Apply mild equalization
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype('uint8')
                
                # Apply equalization to original image
                img_equalized = np.zeros_like(img_array)
                for i in range(3):  # RGB channels
                    channel = (img_array[:, :, i] * 255).astype(np.uint8)
                    img_equalized[:, :, i] = cdf[channel] / 255.0
                
                return img_equalized
            
            return img_array
            
        except Exception as e:
            logger.warning(f"Histogram equalization failed: {e}")
            return img_array
    
    def _image_to_bytes(self, image: Image.Image, format: str = 'JPEG', quality: int = 95) -> bytes:
        """Convert PIL Image to bytes"""
        try:
            image_stream = io.BytesIO()
            image.save(image_stream, format=format, quality=quality, optimize=True)
            return image_stream.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            raise
    
    def extract_image_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """Extract metadata from image for analysis"""
        try:
            image = self._load_image_from_bytes(image_data)
            
            # Basic metadata
            metadata = {
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'size_bytes': len(image_data)
            }
            
            # Calculate image statistics
            img_array = np.array(image)
            metadata.update({
                'mean_brightness': float(np.mean(img_array)),
                'std_brightness': float(np.std(img_array)),
                'min_brightness': int(np.min(img_array)),
                'max_brightness': int(np.max(img_array))
            })
            
            # Color analysis
            if image.mode == 'RGB':
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
                metadata.update({
                    'dominant_color': {
                        'r': float(np.mean(r)),
                        'g': float(np.mean(g)), 
                        'b': float(np.mean(b))
                    }
                })
            
            # Image quality assessment
            metadata['quality_score'] = self._assess_image_quality(img_array)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract image metadata: {e}")
            return {}
    
    def _assess_image_quality(self, img_array: np.ndarray) -> float:
        """Assess image quality for AI processing"""
        try:
            # Calculate sharpness using Laplacian variance
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Calculate brightness distribution
            hist = np.histogram(gray, bins=256, range=[0, 256])[0]
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
            
            # Combine metrics into quality score (0-1)
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize sharpness
            contrast_score = min(contrast / 128.0, 1.0)  # Normalize contrast
            entropy_score = entropy / 8.0  # Normalize entropy
            
            quality_score = (sharpness_score * 0.4 + contrast_score * 0.4 + entropy_score * 0.2)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default quality score
    
    def create_thumbnail(self, image_data: bytes, size: Tuple[int, int] = (128, 128)) -> bytes:
        """Create thumbnail for quick preview"""
        try:
            image = self._load_image_from_bytes(image_data)
            
            # Create thumbnail
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            return self._image_to_bytes(image, quality=80)
            
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            raise

# Usage example and testing
def test_preprocessor():
    """Test the image preprocessor"""
    preprocessor = ImagePreprocessor()
    
    # Create a test image
    test_image = Image.new('RGB', (800, 600), color='red')
    test_bytes = io.BytesIO()
    test_image.save(test_bytes, format='JPEG')
    test_data = test_bytes.getvalue()
    
    try:
        # Test preprocessing
        processed_data = preprocessor.preprocess_image(test_data)
        print(f"Original size: {len(test_data)} bytes")
        print(f"Processed size: {len(processed_data)} bytes")
        
        # Test metadata extraction
        metadata = preprocessor.extract_image_metadata(test_data)
        print(f"Image metadata: {metadata}")
        
        # Test thumbnail creation
        thumbnail_data = preprocessor.create_thumbnail(test_data)
        print(f"Thumbnail size: {len(thumbnail_data)} bytes")
        
        print("✅ Image preprocessor test passed")
        
    except Exception as e:
        print(f"❌ Image preprocessor test failed: {e}")

if __name__ == "__main__":
    test_preprocessor()