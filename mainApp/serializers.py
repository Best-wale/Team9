from rest_framework import serializers
from .models import ProductImage
class ProductImageSerializer(serializers.ModelSerializer):
    image_url = serializers.ImageField(source='image',read_only=True)
    class Meta:
        model = ProductImage
        fields = ['image','image_url']
