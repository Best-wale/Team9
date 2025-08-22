from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ProductImageSerializer
import logging

from .usemodel import predict

logger = logging.getLogger(__name__)

class OCRExtractView(APIView):
    def post(self, request):
        serializer = ProductImageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 1. Save the uploaded image instance
            product_img = serializer.save()

            # 2. Get filesystem path instead of URL
            image_path = product_img.image.path

            # 3. Run your model prediction
            label, score, bbox = predict(image_path)
            authenticity_text = "Suspicious" if label == "fake" else "Genuine"

            # 4. Determine confidence text
            if score > 0.8:
                confidence_text = "High"
            elif score > 0.5:
                confidence_text = "Medium"
            else:
                confidence_text = "Low"

            # 5. Generate analysis detail based on prediction
            analysis_detail = self.generate_analysis_detail(label, score)

            # 6. Return structured response
            return Response({
                'label': authenticity_text,
                'confidence': confidence_text,
                'analysis': analysis_detail,
                'bbox': [float(x) for x in bbox]  # Ensure JSON serializable
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("Model inference failed")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def generate_analysis_detail(self, label, score):
        """Generate a detailed analysis message based on the prediction."""
        if label == "fake":
            if score < 0.5:
                return "The product appears significantly counterfeit."
            elif score < 0.8:
                return "The logo looks slightly off, indicating potential counterfeiting."
            else:
                return "There are some suspicious elements in the product's branding."
        else:
            return "The product appears genuine, but please verify the details."