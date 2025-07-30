workspace "NeuralNetwork"
    configurations {"Debug", "Release"}
    platforms {"Win32", "Win64"}

project "NeuralNetwork"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"

    targetdir "bin/%{cfg.buildcfg}-%{cfg.platform}"
    objdir "bin-int/%{cfg.buildcfg}-%{cfg.platform}"

    files {"Src/**.h", "Src/**.cpp"}

    filter "platforms:Win32"
        architecture "x86"
    filter "platforms:Win64"
        architecture "x64"
    
    filter "configurations:Debug"
        defines {"DEBUG"}
        symbols "on"
    filter "configurations:Release"
        defines {"NDEBUG"}
        optimize "on"
