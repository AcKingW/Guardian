/* 服务器地址 */
var base_url = '175.24.3.72'; //测试服务器
//var base_url = 'http://**********'; //正式服务器


/******************* 配置的拦截器 封装的axios ***********************/
// 创建axios实例
const service = axios.create({
  baseURL: base_url, // api的base_url
  timeout: 120000 // 请求超时时间
})

// request拦截器
service.interceptors.request.use(config => {
  //config.headers['token'] = 66;
  config.headers['Content-Type'] = "application/json";
  // config.headers['id'] = window.sessionStorage.getItem("id");
  // console.log(config.headers)
  return config
}, error => {
  // Do something with request error
  // console.log(error) // for debug
  Promise.reject(error)
})

// respone拦截器
service.interceptors.response.use(
  response => {
      const res = response.data

      return res
  },
  error => {
      console.log(error) // for debug
      Toast('服务器异常, 请稍后重试')

      return Promise.reject(error)
  }
)