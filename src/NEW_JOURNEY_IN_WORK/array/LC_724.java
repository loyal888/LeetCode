package NEW_JOURNEY_IN_WORK.array;

import java.util.HashMap;
import java.util.Map;

public class LC_724 {
    public static int numEquivDominoPairs(int[][] dominoes) {
        HashMap<String, Integer> map = new HashMap<>();
        int ans = 0;
        int row = dominoes.length;
        for (int i = 0; i < row; i++) {
            int[] dominoe = dominoes[i];
            int first = dominoe[0];
            int second = dominoe[1];
            if (second < first) {
                int tmp = first;
                first = second;
                second = tmp;
            }
            String key = first + "" + second;
            if (map.containsKey(key)) {
                int value = map.get(key) + 1;
                map.put(key, value);
            } else {
                map.put(key, 1);
            }
        }
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
                ans+=entry.getValue()*(entry.getValue()-1)/2;
        }

        return ans;
    }

    public static void main(String[] args) {
        numEquivDominoPairs(new int[][]{{1,2},{2,1},{1,2},{3,4},{4,3}});
    }
}
