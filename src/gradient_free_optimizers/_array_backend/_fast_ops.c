/*
 * Fast array operations for GFO's pure Python backend.
 *
 * All functions accept buffer-protocol objects (array.array) via Py_buffer
 * and return bytes objects that can be consumed by array.array.frombytes().
 *
 * Build: compiled as Python C extension module via setuptools.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static PyObject *
fast_vec_add(PyObject *self, PyObject *args)
{
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "y*y*", &a, &b))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf, *bd = (double *)b.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) goto fail;
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = ad[i] + bd[i];

    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return result;
fail:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return NULL;
}

static PyObject *
fast_vec_sub(PyObject *self, PyObject *args)
{
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "y*y*", &a, &b))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf, *bd = (double *)b.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) goto fail;
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = ad[i] - bd[i];

    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return result;
fail:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return NULL;
}

static PyObject *
fast_vec_mul(PyObject *self, PyObject *args)
{
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "y*y*", &a, &b))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf, *bd = (double *)b.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) goto fail;
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = ad[i] * bd[i];

    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return result;
fail:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return NULL;
}

static PyObject *
fast_vec_add_scalar(PyObject *self, PyObject *args)
{
    Py_buffer a;
    double s;
    if (!PyArg_ParseTuple(args, "y*d", &a, &s))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = ad[i] + s;

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_mul_scalar(PyObject *self, PyObject *args)
{
    Py_buffer a;
    double s;
    if (!PyArg_ParseTuple(args, "y*d", &a, &s))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = ad[i] * s;

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_neg(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = -ad[i];

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_clip(PyObject *self, PyObject *args)
{
    Py_buffer a;
    double lo, hi;
    if (!PyArg_ParseTuple(args, "y*dd", &a, &lo, &hi))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++) {
        double v = ad[i];
        rd[i] = v < lo ? lo : (v > hi ? hi : v);
    }

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_exp(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = exp(ad[i]);

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_log(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = log(ad[i]);

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_sqrt(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!result) { PyBuffer_Release(&a); return NULL; }
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (Py_ssize_t i = 0; i < n; i++)
        rd[i] = sqrt(ad[i]);

    PyBuffer_Release(&a);
    return result;
}

static PyObject *
fast_vec_sum(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;
    double s = 0.0;

    for (Py_ssize_t i = 0; i < n; i++)
        s += ad[i];

    PyBuffer_Release(&a);
    return PyFloat_FromDouble(s);
}

static PyObject *
fast_vec_argmax(PyObject *self, PyObject *args)
{
    Py_buffer a;
    if (!PyArg_ParseTuple(args, "y*", &a))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf;

    Py_ssize_t best = 0;
    double best_val = ad[0];
    for (Py_ssize_t i = 1; i < n; i++) {
        if (ad[i] > best_val) {
            best_val = ad[i];
            best = i;
        }
    }

    PyBuffer_Release(&a);
    return PyLong_FromSsize_t(best);
}

static PyObject *
fast_vec_dot(PyObject *self, PyObject *args)
{
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "y*y*", &a, &b))
        return NULL;

    Py_ssize_t n = a.len / (Py_ssize_t)sizeof(double);
    double *ad = (double *)a.buf, *bd = (double *)b.buf;
    double s = 0.0;

    for (Py_ssize_t i = 0; i < n; i++)
        s += ad[i] * bd[i];

    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return PyFloat_FromDouble(s);
}

static PyObject *
fast_mat_mul(PyObject *self, PyObject *args)
{
    Py_buffer a, b;
    int m, k, n;
    if (!PyArg_ParseTuple(args, "y*y*iii", &a, &b, &m, &k, &n))
        return NULL;

    double *ad = (double *)a.buf, *bd = (double *)b.buf;

    PyObject *result = PyBytes_FromStringAndSize(NULL, m * n * sizeof(double));
    if (!result) goto fail;
    double *rd = (double *)PyBytes_AS_STRING(result);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++)
                s += ad[i * k + p] * bd[p * n + j];
            rd[i * n + j] = s;
        }
    }

    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return result;
fail:
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return NULL;
}

static PyMethodDef FastOpsMethods[] = {
    {"vec_add", fast_vec_add, METH_VARARGS, NULL},
    {"vec_sub", fast_vec_sub, METH_VARARGS, NULL},
    {"vec_mul", fast_vec_mul, METH_VARARGS, NULL},
    {"vec_add_scalar", fast_vec_add_scalar, METH_VARARGS, NULL},
    {"vec_mul_scalar", fast_vec_mul_scalar, METH_VARARGS, NULL},
    {"vec_neg", fast_vec_neg, METH_VARARGS, NULL},
    {"vec_clip", fast_vec_clip, METH_VARARGS, NULL},
    {"vec_exp", fast_vec_exp, METH_VARARGS, NULL},
    {"vec_log", fast_vec_log, METH_VARARGS, NULL},
    {"vec_sqrt", fast_vec_sqrt, METH_VARARGS, NULL},
    {"vec_sum", fast_vec_sum, METH_VARARGS, NULL},
    {"vec_argmax", fast_vec_argmax, METH_VARARGS, NULL},
    {"vec_dot", fast_vec_dot, METH_VARARGS, NULL},
    {"mat_mul", fast_mat_mul, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fast_ops_module = {
    PyModuleDef_HEAD_INIT,
    "_fast_ops",
    NULL,
    -1,
    FastOpsMethods
};

PyMODINIT_FUNC
PyInit__fast_ops(void)
{
    return PyModule_Create(&fast_ops_module);
}
