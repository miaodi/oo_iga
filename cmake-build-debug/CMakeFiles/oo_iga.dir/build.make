# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/miaodi/ClionProjects/oo_iga

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/miaodi/ClionProjects/oo_iga/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/oo_iga.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/oo_iga.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/oo_iga.dir/flags.make

CMakeFiles/oo_iga.dir/main.cpp.o: CMakeFiles/oo_iga.dir/flags.make
CMakeFiles/oo_iga.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/oo_iga.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oo_iga.dir/main.cpp.o -c /Users/miaodi/ClionProjects/oo_iga/main.cpp

CMakeFiles/oo_iga.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oo_iga.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/miaodi/ClionProjects/oo_iga/main.cpp > CMakeFiles/oo_iga.dir/main.cpp.i

CMakeFiles/oo_iga.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oo_iga.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/miaodi/ClionProjects/oo_iga/main.cpp -o CMakeFiles/oo_iga.dir/main.cpp.s

CMakeFiles/oo_iga.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/oo_iga.dir/main.cpp.o.requires

CMakeFiles/oo_iga.dir/main.cpp.o.provides: CMakeFiles/oo_iga.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/oo_iga.dir/build.make CMakeFiles/oo_iga.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/oo_iga.dir/main.cpp.o.provides

CMakeFiles/oo_iga.dir/main.cpp.o.provides.build: CMakeFiles/oo_iga.dir/main.cpp.o


CMakeFiles/oo_iga.dir/KnotVector.cpp.o: CMakeFiles/oo_iga.dir/flags.make
CMakeFiles/oo_iga.dir/KnotVector.cpp.o: ../KnotVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/oo_iga.dir/KnotVector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oo_iga.dir/KnotVector.cpp.o -c /Users/miaodi/ClionProjects/oo_iga/KnotVector.cpp

CMakeFiles/oo_iga.dir/KnotVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oo_iga.dir/KnotVector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/miaodi/ClionProjects/oo_iga/KnotVector.cpp > CMakeFiles/oo_iga.dir/KnotVector.cpp.i

CMakeFiles/oo_iga.dir/KnotVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oo_iga.dir/KnotVector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/miaodi/ClionProjects/oo_iga/KnotVector.cpp -o CMakeFiles/oo_iga.dir/KnotVector.cpp.s

CMakeFiles/oo_iga.dir/KnotVector.cpp.o.requires:

.PHONY : CMakeFiles/oo_iga.dir/KnotVector.cpp.o.requires

CMakeFiles/oo_iga.dir/KnotVector.cpp.o.provides: CMakeFiles/oo_iga.dir/KnotVector.cpp.o.requires
	$(MAKE) -f CMakeFiles/oo_iga.dir/build.make CMakeFiles/oo_iga.dir/KnotVector.cpp.o.provides.build
.PHONY : CMakeFiles/oo_iga.dir/KnotVector.cpp.o.provides

CMakeFiles/oo_iga.dir/KnotVector.cpp.o.provides.build: CMakeFiles/oo_iga.dir/KnotVector.cpp.o


CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o: CMakeFiles/oo_iga.dir/flags.make
CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o: ../BsplineBasis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o -c /Users/miaodi/ClionProjects/oo_iga/BsplineBasis.cpp

CMakeFiles/oo_iga.dir/BsplineBasis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oo_iga.dir/BsplineBasis.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/miaodi/ClionProjects/oo_iga/BsplineBasis.cpp > CMakeFiles/oo_iga.dir/BsplineBasis.cpp.i

CMakeFiles/oo_iga.dir/BsplineBasis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oo_iga.dir/BsplineBasis.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/miaodi/ClionProjects/oo_iga/BsplineBasis.cpp -o CMakeFiles/oo_iga.dir/BsplineBasis.cpp.s

CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.requires:

.PHONY : CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.requires

CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.provides: CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.requires
	$(MAKE) -f CMakeFiles/oo_iga.dir/build.make CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.provides.build
.PHONY : CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.provides

CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.provides.build: CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o


CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o: CMakeFiles/oo_iga.dir/flags.make
CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o: ../TensorBsplineBasis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o -c /Users/miaodi/ClionProjects/oo_iga/TensorBsplineBasis.cpp

CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/miaodi/ClionProjects/oo_iga/TensorBsplineBasis.cpp > CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.i

CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/miaodi/ClionProjects/oo_iga/TensorBsplineBasis.cpp -o CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.s

CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.requires:

.PHONY : CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.requires

CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.provides: CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.requires
	$(MAKE) -f CMakeFiles/oo_iga.dir/build.make CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.provides.build
.PHONY : CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.provides

CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.provides.build: CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o


CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o: CMakeFiles/oo_iga.dir/flags.make
CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o: ../PhyTensorBsplineBasis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o -c /Users/miaodi/ClionProjects/oo_iga/PhyTensorBsplineBasis.cpp

CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/miaodi/ClionProjects/oo_iga/PhyTensorBsplineBasis.cpp > CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.i

CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/miaodi/ClionProjects/oo_iga/PhyTensorBsplineBasis.cpp -o CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.s

CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.requires:

.PHONY : CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.requires

CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.provides: CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.requires
	$(MAKE) -f CMakeFiles/oo_iga.dir/build.make CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.provides.build
.PHONY : CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.provides

CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.provides.build: CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o


# Object files for target oo_iga
oo_iga_OBJECTS = \
"CMakeFiles/oo_iga.dir/main.cpp.o" \
"CMakeFiles/oo_iga.dir/KnotVector.cpp.o" \
"CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o" \
"CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o" \
"CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o"

# External object files for target oo_iga
oo_iga_EXTERNAL_OBJECTS =

oo_iga: CMakeFiles/oo_iga.dir/main.cpp.o
oo_iga: CMakeFiles/oo_iga.dir/KnotVector.cpp.o
oo_iga: CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o
oo_iga: CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o
oo_iga: CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o
oo_iga: CMakeFiles/oo_iga.dir/build.make
oo_iga: CMakeFiles/oo_iga.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable oo_iga"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/oo_iga.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/oo_iga.dir/build: oo_iga

.PHONY : CMakeFiles/oo_iga.dir/build

CMakeFiles/oo_iga.dir/requires: CMakeFiles/oo_iga.dir/main.cpp.o.requires
CMakeFiles/oo_iga.dir/requires: CMakeFiles/oo_iga.dir/KnotVector.cpp.o.requires
CMakeFiles/oo_iga.dir/requires: CMakeFiles/oo_iga.dir/BsplineBasis.cpp.o.requires
CMakeFiles/oo_iga.dir/requires: CMakeFiles/oo_iga.dir/TensorBsplineBasis.cpp.o.requires
CMakeFiles/oo_iga.dir/requires: CMakeFiles/oo_iga.dir/PhyTensorBsplineBasis.cpp.o.requires

.PHONY : CMakeFiles/oo_iga.dir/requires

CMakeFiles/oo_iga.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/oo_iga.dir/cmake_clean.cmake
.PHONY : CMakeFiles/oo_iga.dir/clean

CMakeFiles/oo_iga.dir/depend:
	cd /Users/miaodi/ClionProjects/oo_iga/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/miaodi/ClionProjects/oo_iga /Users/miaodi/ClionProjects/oo_iga /Users/miaodi/ClionProjects/oo_iga/cmake-build-debug /Users/miaodi/ClionProjects/oo_iga/cmake-build-debug /Users/miaodi/ClionProjects/oo_iga/cmake-build-debug/CMakeFiles/oo_iga.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/oo_iga.dir/depend

