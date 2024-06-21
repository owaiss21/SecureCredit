from django.shortcuts import render
import joblib
from django.shortcuts import render
from django.http import JsonResponse

def Home(request):
    return render(request, 'home.html')

# Load the model
model = joblib.load('fraud_detection_model.pkl')

def predict(request):
    if request.method == 'POST':
        print(request.POST)
        ratio_to_median_purchase_price = float(request.POST.get('median'))
        repeat_retailer = float(request.POST.get('price'))
        distance_from_home = float(request.POST.get('distance'))
        distance_from_last_transaction = float(request.POST.get('purchase_method'))
        used_chip = float(request.POST.get('field1'))
        used_pin_number = float(request.POST.get('field2'))
        online_order = float(request.POST.get('field3'))
        

        prediction = model.predict([[ratio_to_median_purchase_price, repeat_retailer, distance_from_home, distance_from_last_transaction,used_chip,used_pin_number,online_order]])[0]
        print(prediction)
        result = 'Fraud' if prediction == 1 else 'Not Fraud'

        return JsonResponse({'result': result})

    return render(request, 'form.html')
