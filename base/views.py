from django.shortcuts import render
from .recognizer import recognize

# Create your views here.
def home(request):
    return render(request, 'home.html')


def predict(request):
	if request.method == 'POST':
		best, others, img = recognize(request.FILES['image'])
		context = {
			'best': best,
			'others': others,
			'img': img
		}
		return render(request, 'predict.html', context)
