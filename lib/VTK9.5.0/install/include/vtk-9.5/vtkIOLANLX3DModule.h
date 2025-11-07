
#ifndef VTKIOLANLX3D_EXPORT_H
#define VTKIOLANLX3D_EXPORT_H

#ifdef VTKIOLANLX3D_STATIC_DEFINE
#  define VTKIOLANLX3D_EXPORT
#  define VTKIOLANLX3D_NO_EXPORT
#else
#  ifndef VTKIOLANLX3D_EXPORT
#    ifdef IOLANLX3D_EXPORTS
        /* We are building this library */
#      define VTKIOLANLX3D_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define VTKIOLANLX3D_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef VTKIOLANLX3D_NO_EXPORT
#    define VTKIOLANLX3D_NO_EXPORT 
#  endif
#endif

#ifndef VTKIOLANLX3D_DEPRECATED
#  define VTKIOLANLX3D_DEPRECATED __declspec(deprecated)
#endif

#ifndef VTKIOLANLX3D_DEPRECATED_EXPORT
#  define VTKIOLANLX3D_DEPRECATED_EXPORT VTKIOLANLX3D_EXPORT VTKIOLANLX3D_DEPRECATED
#endif

#ifndef VTKIOLANLX3D_DEPRECATED_NO_EXPORT
#  define VTKIOLANLX3D_DEPRECATED_NO_EXPORT VTKIOLANLX3D_NO_EXPORT VTKIOLANLX3D_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VTKIOLANLX3D_NO_DEPRECATED
#    define VTKIOLANLX3D_NO_DEPRECATED
#  endif
#endif

/* VTK-HeaderTest-Exclude: vtkIOLANLX3DModule.h */

/* Include ABI Namespace */
#include "vtkABINamespace.h"

#endif /* VTKIOLANLX3D_EXPORT_H */
