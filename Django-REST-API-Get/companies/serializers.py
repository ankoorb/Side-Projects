from rest_framework import serializers
from .models import Stock

class StockSerializer(serializers.ModelSerializer):

    class Meta:
        model = Stock
        #fields = ('ticker', 'volume') # From models.py Stock
        fields = '__all__' # Return all