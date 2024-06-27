from django.shortcuts import render
import joblib
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

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

    data = pd.read_csv("fraud_data.csv")

    # Drop missing values and convert 'fraud' column to integer type
    data.dropna(inplace=True)
    data['fraud'] = data['fraud'].astype(int)
    payment_methods = ['repeat_retailer', 'online_order', 'used_pin_number', 'used_chip']

    # Prepare the plot
    plots = []
    for method in payment_methods:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=method, hue='fraud', palette='viridis')
        plt.title(f'{method.replace("_", " ").title()} Counts')
        plt.xlabel(method.replace("_", " ").title())
        plt.ylabel('Count')
        plt.legend(title='FRAUD', labels=['Non-Fraudulent', 'Fraudulent'])

        # Save plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode plot image to base64
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        plots.append(graphic)

        plt.close()

    context = {'plots': plots}
    # return render(request, 'fraud_analysis.html', context)
    return render(request, 'form.html', context)
