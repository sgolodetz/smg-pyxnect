#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <opencv2/opencv.hpp>

#pragma warning(disable:4101 4244 4267 4700)
#include <xnect.hpp>
#pragma warning(default:4101 4244 4267 4700)

PYBIND11_MODULE(pyxnect, m)
{
  // CLASSES

  py::class_<XNECT>(m, "XNect")
    .def(
      py::init<std::string>(),
      py::arg("config_file") = "../../data/FullBodyTracker/",
      py::call_guard<py::gil_scoped_release>()
    )
    .def(
      "get_joint3d_ik",
      [](XNECT& self, int person, int joint)
      {
        cv::Vec3f rawResult = self.getJoint3DIK(person, joint);
        auto result = py::array_t<double>(3);
        py::buffer_info buf = result.request();
        double *ptr = (double*)buf.ptr;
        for(int i = 0; i < 3; ++i) ptr[i] = rawResult[i];
        return result;
      },
      py::call_guard<py::gil_scoped_release>()
    )
    .def("get_joint3d_parent", &XNECT::getJoint3DParent, py::call_guard<py::gil_scoped_release>())
    .def("get_num_of_3d_joints", &XNECT::getNumOf3DJoints, py::call_guard<py::gil_scoped_release>())
    .def("get_num_of_people", &XNECT::getNumOfPeople, py::call_guard<py::gil_scoped_release>())
    .def(
      "get_person_colour",
      [](XNECT& self, int p)
      {
        cv::Vec3b rawResult = self.getPersonColor(p);
        auto result = py::array_t<int>(3);
        py::buffer_info buf = result.request();
        int *ptr = (int*)buf.ptr;
        for(int i = 0; i < 3; ++i) ptr[i] = rawResult[i];
        return result;
      },
      py::call_guard<py::gil_scoped_release>()
    )
    .def("is_person_active", &XNECT::isPersonActive, py::call_guard<py::gil_scoped_release>())
    .def(
      "process_image",
      [](XNECT& self, py::array_t<uint8_t>& img)
      {
        py::buffer_info buf = img.request();
        cv::Mat mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
        self.processImg(mat);
      },
      py::call_guard<py::gil_scoped_release>()
    )
    .def(
      "project_with_intrinsics",
      [](XNECT& self, py::array_t<float>& point)
      {
        py::buffer_info pointBuf = point.request();
        float *pointPtr = (float*)pointBuf.ptr;
        cv::Vec2f rawResult = self.ProjectWithIntrinsics(cv::Vec3f((float*)pointBuf.ptr));

        auto result = py::array_t<double>(2);
        py::buffer_info resultBuf = result.request();
        double *resultPtr = (double*)resultBuf.ptr;
        for(int i = 0; i < 2; ++i) resultPtr[i] = rawResult[i];

        return result;
      },
      py::call_guard<py::gil_scoped_release>()
    )
  ;
}
