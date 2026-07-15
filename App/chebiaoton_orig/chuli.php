<?php
header('Access-Control-Allow-Origin: *'); 
header('Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept'); 
$servername="localhost";
$user="root";
$password="";
$conn=mysqli_connect($servername,$user,$password);
if(!$conn)
{
	die("connect_failed:".mysqli_error());
}
else{
	if($_POST["act"]=='"select"')
	{
		$sql="select * from cs.user";
		$arr=mysqli_query($conn,$sql);
		$rows=mysqli_affected_rows($conn);
		for($i=1;$i<=$rows;$i++)
		{
			$row=mysqli_fetch_array($arr);
			echo '<tr id='.$row["id"].'><td>'.$row["id"].'</td><td>'.$row["name"].'</td><td>'.$row["sex"].'</td><td>'.$row["result"].'</td><td><input type="button"class="btn btn-sm btn-default"value="删除"name="'.$row["id"].'"onclick="remove('.$row["id"].')"/></td></tr>';
			echo '<tr style="font-size:25px;"><td colspan="4"><ul><li>请假单的编号是:'.$row["id"].'</li><br/><li>请假者的名字是:'.$row["name"].'</li><br/><li>请假者的学号是:'.$row["xuehao"].'</li><br/><li>请假者的手机是:'.$row["phone"].'</li><br/><li>请假者的班级是:'.$row["sex"].'</li><br/><li>请假者的去向是:'.$row["fangxiang"].'</li><br/><li>请假者请假的时间是:'.$row["time"].'</li></br><li>请假者请假的原因是:'.$row["reason"].'</li><br/><li>紧急程度是:'.$row["chengdu"].'</li><br/><li>家长是否知情：'.$row["zhiqing"].'</li></ul></td></tr>';
		}
	}
	if($_POST["act"]=='1')
	{
		$sql0='select id from cs.user order by id desc limit 1';
		$arr0=mysqli_query($conn,$sql0);
		$row0=mysqli_fetch_array($arr0);
		$id=$row0[0]+1;
		$name=$_POST["name"];
		$time1=$_POST["time"];
		$time=intval($time1);
		$chengdu=$_POST["chengdu"];
		$class=$_POST["class"];
		$reason=$_POST["reason"];
		$phone=$_POST["phone"];
		$xuehao=$_POST["xuehao"];
		$zhiqing=$_POST["zhiqing"];
		$fangxiang=$_POST["fangxiang"];
		$sql='insert into cs.user values('.$id.',"'.$name.'",'.$xuehao.','.$phone.',"'.$class.'","'.$fangxiang.'","'.$time1.'",'.$chengdu.',"'.$zhiqing.'","'.$reason.'","审批中")';
		if(mysqli_query($conn,$sql))
		{
			echo '提交成功';																																																																											
		}
		else
		{
			echo '提交错误！请检查你的输入格式。';	
		}
		header("Location:qingjia.html"); 
		exit;
	}
}
?>