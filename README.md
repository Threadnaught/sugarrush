# Sugarrush

Dead simple hyperparameter iterator and logger. You give it configs and reports, it logs how your training is going and allows you to compare between runs at the end. See `sugarrush-example.py` to see how easy it is to use. You can pass it info as python native types, np arrays, or torch arrays and it will automatically detach them and store them on the CPU.