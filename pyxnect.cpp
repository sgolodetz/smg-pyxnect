#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
namespace py = pybind11;

// TODO
//#include <xnect.hpp>

PYBIND11_MODULE(pyxnect, m)
{
}

#if 0
#include <Python.h>

#include <opencv2/core.hpp>

PYBIND11_MODULE(pyopencv, m)
{
  // CLASSES

  py::class_<cv::Mat1b>(m, "CVMat1b", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1b& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(unsigned char),
        pybind11::format_descriptor<unsigned char>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(unsigned char) * img.cols, sizeof(unsigned char) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1b {
      return cv::Mat1b::zeros(rows, cols);
    }, py::call_guard<py::gil_scoped_release>())
  ;

  py::class_<cv::Mat1d>(m, "CVMat1d", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1d& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(double),
        pybind11::format_descriptor<double>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(double) * img.cols, sizeof(double) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1d { return cv::Mat1d::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat1f>(m, "CVMat1f", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1f& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(float),
        pybind11::format_descriptor<float>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(float) * img.cols, sizeof(float) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1f { return cv::Mat1f::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat1i>(m, "CVMat1i", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1i& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(int),
        pybind11::format_descriptor<int>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(int) * img.cols, sizeof(int) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1i { return cv::Mat1i::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat3b>(m, "CVMat3b", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat3b& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(unsigned char),
        pybind11::format_descriptor<unsigned char>::format(),
        3,
        { img.rows, img.cols, 3 },
        { sizeof(unsigned char) * 3 * img.cols, sizeof(unsigned char) * 3, sizeof(unsigned char) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat3b { return cv::Mat3b::zeros(rows, cols); }, "")
  ;
}
#endif
