$workspace="projects/yk-wetland-mapping/assets/test-workspace"
earthengine rm -r $workspace 
# -- re create the init folder
earthengine create folder $workspace