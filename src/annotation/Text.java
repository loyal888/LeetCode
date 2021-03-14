package annotation;

import java.lang.annotation.Annotation;

import java.lang.reflect.Field;

import java.lang.reflect.Method;

public class Text {
    Annotation[] annotation = null;

    public static void main(String[] args) throws ClassNotFoundException {
        new Text().getAnnotation();

    }

    public void getAnnotation() throws ClassNotFoundException {
        Class stu = Class.forName("annotation.Student");//静态加载类

        boolean isEmpty = stu.isAnnotationPresent(annotation.Annotation_my.class);//判断stu是不是使用了我们刚才定义的注解接口if(isEmpty){
        annotation = stu.getAnnotations();//获取注解接口中的

        for (Annotation a : annotation) {
            Annotation_my my = (Annotation_my) a;//强制转换成Annotation_my类型

            System.out.println(stu + ":\n" + my.name() + " say: " + my.say() + " my age: " + my.age());

        }

    }
}