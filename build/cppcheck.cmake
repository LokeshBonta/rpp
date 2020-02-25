
        file(GLOB_RECURSE GSRCS  /home/mvx/git/AMDRPP/addkernels/*.cpp /home/mvx/git/AMDRPP/addkernels/*.hpp /home/mvx/git/AMDRPP/addkernels/*.cxx /home/mvx/git/AMDRPP/addkernels/*.c /home/mvx/git/AMDRPP/addkernels/*.h /home/mvx/git/AMDRPP/include/*.cpp /home/mvx/git/AMDRPP/include/*.hpp /home/mvx/git/AMDRPP/include/*.cxx /home/mvx/git/AMDRPP/include/*.c /home/mvx/git/AMDRPP/include/*.h /home/mvx/git/AMDRPP/src/*.cpp /home/mvx/git/AMDRPP/src/*.hpp /home/mvx/git/AMDRPP/src/*.cxx /home/mvx/git/AMDRPP/src/*.c /home/mvx/git/AMDRPP/src/*.h /home/mvx/git/AMDRPP/test/*.cpp /home/mvx/git/AMDRPP/test/*.hpp /home/mvx/git/AMDRPP/test/*.cxx /home/mvx/git/AMDRPP/test/*.c /home/mvx/git/AMDRPP/test/*.h)
        set(CPPCHECK_COMMAND
            CPPCHECK_EXE-NOTFOUND
            -q
            # -v
            # --report-progress
            --force
            --cppcheck-build-dir=/home/mvx/git/AMDRPP/build/cppcheck-build
            --platform=native
            --template=gcc
            --error-exitcode=1
            -j 64
             -DMIOPEN_USE_MIOPENGEMM=1
            
             -I/home/mvx/git/AMDRPP/include -I/home/mvx/git/AMDRPP/build/include -I/home/mvx/git/AMDRPP/src/include
            --enable=all
            --inline-suppr
            --suppressions-list=/home/mvx/git/AMDRPP/build/cppcheck-supressions
             ${GSRCS}
        )
        string(REPLACE ";" " " CPPCHECK_SHOW_COMMAND "${CPPCHECK_COMMAND}")
        message("${CPPCHECK_SHOW_COMMAND}")
        execute_process(
            COMMAND ${CPPCHECK_COMMAND}
            WORKING_DIRECTORY /home/mvx/git/AMDRPP
            RESULT_VARIABLE RESULT
        )
        if(NOT RESULT EQUAL 0)
            message(FATAL_ERROR "Cppcheck failed")
        endif()
