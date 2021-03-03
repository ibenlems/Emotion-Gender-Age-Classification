function helper() {
	green=$(tput setaf 2)
	red=$(tput setaf 1)
	reset=$(tput sgr0)
	echo -e "${red}All commands should be executed at the same directory as README file${reset}"
	echo -e "\n${green}detector image input_path output_path${reset} : draw bounding boxes around faces within image and detect the emotion, age and gender.${green}Hit Enter button to quit${reset}\n\n\tThe output_path should include the image name and format \n \t Example : 'detector image data/face.jpeg data/output.jpeg' \n
${green}detector video input_path output_path stride[optional(integer),default=1]${reset} : draw bounding boxes around faces every ${green}stride${reset} frames within video and detect the emotion, age and gender.\n\n\tThe output_path should include the video name and format (avi format is privileged) \n \t Example : 'detector video data/china.avi data/output.avi'\n
${green}detector webcam${reset} : draw bounding boxes around faces within webcam frame and detect the emotion, age and gender.${green}Hit 'q' on the keyboard to quit${reset}\n\n \t Example : 'detector webcam'
	"
}

function detector() {

if [ $# -eq 0 ] 
then
    echo "No arguments supplied, Try 'detector help' for more information "
elif [ $1 == "help" ]
then
	helper
elif [ $1 == "webcam" ]
then 
	python3 ./Code/detector.py $1 2>/dev/null
else
	if [ "$2" -a "$3" ]
	then
		python3 ./Code/detector.py $1 $2 $3 2>/dev/null
	else
		echo "Wrong arguments were given, Try 'detector help' for more information "
    	fi
fi
}
