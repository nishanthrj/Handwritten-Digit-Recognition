submitBtn = document.querySelector('#submit-btn')
fileField = document.querySelector('#file-field')
form = document.querySelector('#upload-form')

submitBtn.addEventListener('click', (e) => {
	e.preventDefault();
	fileField.click()
})


fileField.addEventListener('change', (e) => {
	e.preventDefault();
	form.submit()
})
