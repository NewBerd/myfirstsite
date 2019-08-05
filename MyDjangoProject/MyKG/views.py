from django.shortcuts import render

# Create your views here.

from django import forms
from .models import Profile
# Create your views here.

class ProfileForm(forms.Form):
   name = forms.CharField(max_length = 100)
   picture = forms.ImageField()


def saveProfile(request):


    if request.method == "POST":
        # Get the posted form
        MyProfileForm = ProfileForm(request.POST, request.FILES)

        if MyProfileForm.is_valid():
            profile = Profile()
            profile.name = MyProfileForm.cleaned_data["name"]
            profile.picture = MyProfileForm.cleaned_data["picture"]
            profile.save()

    else:
        MyProfileForm = ProfileForm()

    return render(request, 'saved.html', {"form":MyProfileForm})

def showImages(request):
    objs = Profile.objects.all()
    print objs
    return  render(request,"list.html",{"pics":objs})