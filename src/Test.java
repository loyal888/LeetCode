import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Proxy;
import java.util.*;

public class Test {
    public static void main(String[] args) {
        int[] a = new int[]{
                1, 2
        };
        for (int aa : a) {
            System.out.println(aa);
        }
        int i = 0;
        while (i < a.length) {
            System.out.println(a[i]);
            i++;
        }
    }
}
