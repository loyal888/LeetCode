import org.omg.CORBA.INTERNAL;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LeetCode {

    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public static int myAtoi(String str) {
        // 去除空格
        str = str.trim();
        // 保证为整数
        boolean matches = str.matches("[-+]?\\d+.*");
        int ans = 0;
        if (matches) {
            // 截取字符串
            String pattern = "[-+]?\\d+";
            Pattern p = Pattern.compile(pattern);
            Matcher matcher = p.matcher(str);
            if (matcher.find()) {
                str = matcher.group();
            }
            // 判断第一位是否为正负号
            char c = str.charAt(0);
            int pos = 0;
            int nag = 0;
            switch (c) {
                case '+':
                    pos = 1;
                    break;
                case '-':
                    nag = 1;
                    break;
                default:
                    break;
            }
            // 第一位为正数
            if (pos == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            } else if (nag == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (-ans < Integer.MIN_VALUE / 10 || (-ans == Integer.MIN_VALUE / 10 && res > 8)) {
                        return -2147483648;
                    }
                    ans = ans * 10 + res;
                }
                ans = ans * (-1);
            } else {
                for (int i = 0; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            }
            return ans;
        }
        return 0;
    }

    public static String intToRoman(int num) {
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] romans = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int index = 0;
        StringBuilder sb = new StringBuilder();
        while (index < nums.length) {

            while (num >= nums[index]) {
                sb.append(romans[index]);
                num = num - nums[index];
            }
            index++;
        }
        return sb.toString();

    }

    public static int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("IV", 4);
        map.put("V", 5);
        map.put("IX", 9);
        map.put("X", 10);
        map.put("XL", 40);
        map.put("L", 50);
        map.put("XC", 90);
        map.put("C", 100);
        map.put("CD", 400);
        map.put("D", 500);
        map.put("CM", 900);
        map.put("M", 1000);

        int ans = 0;
        for (int i = 0; i < s.length(); ) {
            if (i + 1 < s.length() && map.containsKey(s.substring(i, i + 2))) {
                ans += map.get(s.substring(i, i + 2));
                i += 2;
            } else {
                ans += map.get(s.substring(i, i + 1));
                i++;
            }
        }
        return ans;
    }

    /**
     * 14. 最长公共前缀
     * 标签：链表
     * 当字符串数组长度为 0 时则公共前缀为空，直接返回
     * 令最长公共前缀 ans 的值为第一个字符串，进行初始化
     * 遍历后面的字符串，依次将其与 ans 进行比较，两两找出公共前缀，最终结果即为最长公共前缀
     * 如果查找过程中出现了 ans 为空的情况，则公共前缀不存在直接返回
     * 时间复杂度：O(s)O(s)，s 为所有字符串的长度之和
     *
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0)
            return "";
        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j))
                    break;
            }
            ans = ans.substring(0, j);
            if (ans.equals(""))
                return ans;
        }
        return ans;
    }

    /**
     * 15. 三数之和
     *
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList();
        int len = nums.length;
        if (nums == null || len < 3) return ans;
        Arrays.sort(nums); // 排序
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) break;
            // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 去重
            int L = i + 1;
            int R = len - 1;
            while (L < R) {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[L], nums[R]));
                    while (L < R && nums[L] == nums[L + 1]) L++; // 去重
                    while (L < R && nums[R] == nums[R - 1]) R--; // 去重
                    L++;
                    R--;
                } else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }
        return ans;
    }

    /**
     * 16. 最接近的三数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return -1;
        }
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(target - sum) < Math.abs(target - ans)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    /**
     * 17. 电话号码的字母组合
     *
     * @param digits
     * @return
     */
    public static List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if (digits.isEmpty()) return ans;
        String[] mapping = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
//            int x = Character.getNumericValue(digits.charAt(i));
            int x = Integer.valueOf(digits.charAt(i) + "");
            while (ans.peek().length() == i) {
                String t = ans.remove();
                for (char s : mapping[x].toCharArray())
                    ans.add(t + s);
            }
        }
        return ans;
    }

    /**
     * 18. 四数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public static List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums == null || nums.length < 4) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int start = j + 1, end = nums.length - 1;

                while (start < end) {
                    int temp = nums[i] + nums[j] + nums[start] + nums[end];
                    if (target == temp) {
                        ArrayList<Integer> tList = new ArrayList<>();
                        tList.add(nums[i]);
                        tList.add(nums[j]);
                        tList.add(nums[start]);
                        tList.add(nums[end]);
                        ;
                        ans.add(tList);
                        start++;
                        end--;
                    } else if (target > temp) {
                        start++;

                    } else {
                        end--;
                    }
                }

            }
        }
        return new ArrayList<>(new HashSet<>(ans));
    }

    /**
     * 19.删除链表的倒数第N个节点
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode p = new ListNode(0);
        p.next = head;
        ListNode first = p;
        ListNode second = p;
        // 提前移动 n+1步
        for (int i = 1; i <= n + 1; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return p.next;
    }

    /**
     * 20. 有效的括号
     *
     * @param s
     * @return
     */
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        HashMap<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            if (map.containsKey(c)) {
                Character value = stack.empty() ? '#' : stack.pop();
                if (!value.equals(map.get(c))) {
                    return false;
                }
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    public static  ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }

        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    public static void main(String[] args) {
        int[] nums = {1, 0, -1, 0, -2, 2};
        ListNode listNode = new ListNode(1);
        ListNode listNode1 = new ListNode(2);
        ListNode listNode2 = new ListNode(3);
        ListNode listNode3 = new ListNode(4);
        ListNode listNode4 = new ListNode(5);

        listNode.next = listNode2;
        listNode2.next = listNode4;
        listNode1.next = listNode3;

        mergeTwoLists(listNode,listNode1);

    }


}
